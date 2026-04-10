#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit a local baseline matrix suite and explain why variants differ.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


CORE_METRICS: Tuple[str, ...] = (
    "doc_role_micro_f1",
    "doc_instance_micro_f1",
    "doc_combination_micro_f1",
    "doc_event_type_micro_f1",
    "strict_f1",
    "relaxed_f1",
    "type_f1",
    "schema_compliance_rate",
    "hallucination_rate",
    "avg_tokens_per_sample",
    "total_tokens",
    "wall_clock_seconds",
    "samples_per_second",
    "f1_per_1k_tokens",
    "f1_per_minute",
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def extract_event_types(events: Iterable[Mapping[str, Any]]) -> List[str]:
    seen: List[str] = []
    for event in events:
        event_type = str(event.get("event_type", "")).strip()
        if event_type and event_type not in seen:
            seen.append(event_type)
    return seen


def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _tuple_key(items: Sequence[str]) -> str:
    return " | ".join(items) if items else "<none>"


def summarize_result_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    parse_success = 0
    total_gold_events = 0
    total_pred_events = 0
    total_gold_types = 0
    total_schema_size = 0
    total_stage1_predicted_types = 0
    stage1_rows = 0
    stage1_empty = 0
    stage1_gold_covered = 0
    stage1_exact = 0
    stage1_missed = 0
    stage1_overpredicted = 0
    fewshot_example_counter: Counter[str] = Counter()
    fewshot_combo_counter: Counter[str] = Counter()

    for row in rows:
        if bool(row.get("parse_success")):
            parse_success += 1

        gold_events = row.get("ground_truth") or []
        pred_events = row.get("prediction_canonical") or row.get("prediction") or []
        total_gold_events += len(gold_events)
        total_pred_events += len(pred_events)

        gold_types = set(extract_event_types(gold_events))
        total_gold_types += len(gold_types)

        prompt_meta = row.get("prompt_meta") or {}
        example_ids = [str(item) for item in prompt_meta.get("fewshot_example_ids", []) if str(item)]
        for example_id in example_ids:
            fewshot_example_counter[example_id] += 1
        if example_ids:
            fewshot_combo_counter[_tuple_key(example_ids)] += 1

        stage_meta = row.get("stage_meta") or {}
        stage_mode = str(stage_meta.get("stage_mode", "single_pass"))
        schema_types = [str(item) for item in stage_meta.get("stage2_schema_event_types", []) if str(item)]
        total_schema_size += len(schema_types)

        if stage_mode != "two_stage":
            continue

        stage1_rows += 1
        predicted_types = {
            str(item).strip()
            for item in stage_meta.get("stage1_predicted_event_types", [])
            if str(item).strip()
        }
        total_stage1_predicted_types += len(predicted_types)

        if not predicted_types:
            stage1_empty += 1
        if gold_types and gold_types.issubset(predicted_types):
            stage1_gold_covered += 1
        elif gold_types:
            stage1_missed += 1
        if predicted_types == gold_types:
            stage1_exact += 1
        if predicted_types - gold_types:
            stage1_overpredicted += 1

    top_examples = [
        {"example_id": example_id, "count": count}
        for example_id, count in fewshot_example_counter.most_common(10)
    ]
    top_combos = [
        {"example_ids": combo, "count": count}
        for combo, count in fewshot_combo_counter.most_common(10)
    ]

    return {
        "total_rows": total,
        "parse_success_rate": safe_div(parse_success, total),
        "avg_gold_events": safe_div(total_gold_events, total),
        "avg_predicted_events": safe_div(total_pred_events, total),
        "avg_gold_event_types": safe_div(total_gold_types, total),
        "avg_schema_event_types": safe_div(total_schema_size, total),
        "stage1_rows": stage1_rows,
        "stage1_empty_rate": safe_div(stage1_empty, stage1_rows),
        "stage1_gold_coverage_rate": safe_div(stage1_gold_covered, stage1_rows),
        "stage1_exact_match_rate": safe_div(stage1_exact, stage1_rows),
        "stage1_miss_rate": safe_div(stage1_missed, stage1_rows),
        "stage1_overprediction_rate": safe_div(stage1_overpredicted, stage1_rows),
        "avg_stage1_predicted_types": safe_div(total_stage1_predicted_types, stage1_rows),
        "fewshot_unique_example_ids": len(fewshot_example_counter),
        "fewshot_unique_combinations": len(fewshot_combo_counter),
        "fewshot_top_examples": top_examples,
        "fewshot_top_combinations": top_combos,
    }


def build_variant_record(name: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    run_dir = Path(str(payload["run_dir"]))
    result_path = run_dir / "eval_results.jsonl"
    row_summary = summarize_result_rows(load_jsonl(result_path))
    metrics = {
        metric_name: payload.get("metrics", {}).get(metric_name)
        for metric_name in CORE_METRICS
    }
    return {
        "variant": name,
        "run_dir": str(run_dir),
        "summary_file": str(payload.get("summary_file", "")),
        "prompt_variant": str(payload.get("prompt_variant", "zeroshot")),
        "stage_mode": str(payload.get("stage_mode", "single_pass")),
        "use_fewshot": bool(payload.get("use_fewshot", False)),
        "batch_size": int(payload.get("batch_size", 0)),
        "metrics": metrics,
        "row_diagnostics": row_summary,
    }


def build_pairwise_deltas(
    variants: Mapping[str, Mapping[str, Any]],
    *,
    reference_variant: str,
) -> Dict[str, Dict[str, float]]:
    ref = variants[reference_variant]["metrics"]
    deltas: Dict[str, Dict[str, float]] = {}
    for name, payload in variants.items():
        if name == reference_variant:
            continue
        current = payload["metrics"]
        metric_delta: Dict[str, float] = {}
        for metric_name in (
            "doc_role_micro_f1",
            "doc_instance_micro_f1",
            "doc_combination_micro_f1",
            "doc_event_type_micro_f1",
            "strict_f1",
            "relaxed_f1",
            "schema_compliance_rate",
            "hallucination_rate",
            "avg_tokens_per_sample",
            "wall_clock_seconds",
        ):
            if ref.get(metric_name) is None or current.get(metric_name) is None:
                continue
            metric_delta[metric_name] = float(current[metric_name]) - float(ref[metric_name])
        deltas[name] = metric_delta
    return deltas


def derive_findings(audit: Mapping[str, Any], *, reference_variant: str) -> List[str]:
    findings: List[str] = []
    variants = audit["variants"]

    ref = variants[reference_variant]
    sp_few = variants.get("single_pass_fewshot")
    ts_zero = variants.get("two_stage_zeroshot")
    ts_few = variants.get("two_stage_fewshot")

    if sp_few:
        ref_role = float(ref["metrics"]["doc_role_micro_f1"])
        few_role = float(sp_few["metrics"]["doc_role_micro_f1"])
        token_ratio = safe_div(
            float(sp_few["metrics"]["avg_tokens_per_sample"]),
            float(ref["metrics"]["avg_tokens_per_sample"]),
        )
        combo_count = int(sp_few["row_diagnostics"]["fewshot_unique_combinations"])
        findings.append(
            "single_pass fewshot is weaker than single_pass zeroshot while using "
            f"{token_ratio:.2f}x tokens/sample; dynamic retrieval only produced "
            f"{combo_count} unique exemplar combinations across the run."
        )

    if ts_zero:
        coverage = float(ts_zero["row_diagnostics"]["stage1_gold_coverage_rate"])
        avg_stage1_types = float(ts_zero["row_diagnostics"]["avg_stage1_predicted_types"])
        avg_gold_types = float(ts_zero["row_diagnostics"]["avg_gold_event_types"])
        findings.append(
            "two_stage zeroshot loses recall before extraction: "
            f"stage1 gold coverage={coverage:.4f}, avg predicted types={avg_stage1_types:.2f}, "
            f"avg gold event types={avg_gold_types:.2f}."
        )

    if ts_few:
        coverage = float(ts_few["row_diagnostics"]["stage1_gold_coverage_rate"])
        exact = float(ts_few["row_diagnostics"]["stage1_exact_match_rate"])
        combo_count = int(ts_few["row_diagnostics"]["fewshot_unique_combinations"])
        findings.append(
            "two_stage fewshot compounds two bottlenecks: "
            f"stage1 gold coverage={coverage:.4f}, exact match={exact:.4f}, "
            f"fewshot combinations={combo_count}."
        )

    best_variant = max(
        variants.items(),
        key=lambda item: float(item[1]["metrics"]["doc_role_micro_f1"]),
    )[0]
    findings.append(f"best variant by doc_role_micro_f1 is {best_variant}.")
    return findings


def render_markdown(audit: Mapping[str, Any], *, reference_variant: str) -> str:
    lines = [
        "# Baseline Matrix Audit",
        "",
        f"- Suite: `{audit['suite']}`",
        f"- Reference variant: `{reference_variant}`",
        "",
        "## Variant Metrics",
        "",
        "| Variant | doc_role | doc_instance | doc_combination | doc_event_type | strict_f1 | relaxed_f1 | schema | hallucination | avg_tokens | wall_clock_s |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, payload in audit["variants"].items():
        metrics = payload["metrics"]
        lines.append(
            "| "
            + f"{name} | "
            + f"{metrics['doc_role_micro_f1']:.4f} | "
            + f"{metrics['doc_instance_micro_f1']:.4f} | "
            + f"{metrics['doc_combination_micro_f1']:.4f} | "
            + f"{metrics['doc_event_type_micro_f1']:.4f} | "
            + f"{metrics['strict_f1']:.4f} | "
            + f"{metrics['relaxed_f1']:.4f} | "
            + f"{metrics['schema_compliance_rate']:.4f} | "
            + f"{metrics['hallucination_rate']:.4f} | "
            + f"{metrics['avg_tokens_per_sample']:.2f} | "
            + f"{metrics['wall_clock_seconds']:.2f} |"
        )

    lines.extend(["", "## Key Findings", ""])
    for finding in audit["findings"]:
        lines.append(f"- {finding}")

    lines.extend(["", "## Stage Diagnostics", ""])
    for name, payload in audit["variants"].items():
        diag = payload["row_diagnostics"]
        lines.append(f"### {name}")
        lines.append(
            f"- parse_success_rate={diag['parse_success_rate']:.4f}, "
            f"avg_gold_events={diag['avg_gold_events']:.2f}, "
            f"avg_predicted_events={diag['avg_predicted_events']:.2f}"
        )
        lines.append(
            f"- stage1_gold_coverage_rate={diag['stage1_gold_coverage_rate']:.4f}, "
            f"stage1_exact_match_rate={diag['stage1_exact_match_rate']:.4f}, "
            f"avg_stage1_predicted_types={diag['avg_stage1_predicted_types']:.2f}"
        )
        lines.append(
            f"- fewshot_unique_example_ids={diag['fewshot_unique_example_ids']}, "
            f"fewshot_unique_combinations={diag['fewshot_unique_combinations']}"
        )
    lines.append("")
    return "\n".join(lines)


def audit_suite(
    suite_summary_path: Path,
    *,
    reference_variant: str = "single_pass_zeroshot",
) -> Dict[str, Any]:
    summary = load_json(suite_summary_path)
    variants_payload = summary.get("variants") or {}
    if reference_variant not in variants_payload:
        raise ValueError(f"reference variant not found in suite summary: {reference_variant}")

    variants = {
        name: build_variant_record(name, payload)
        for name, payload in variants_payload.items()
    }
    return {
        "suite": summary.get("suite"),
        "model": summary.get("model"),
        "gpu": summary.get("gpu"),
        "reference_variant": reference_variant,
        "variants": variants,
        "pairwise_deltas": build_pairwise_deltas(variants, reference_variant=reference_variant),
        "findings": derive_findings({"variants": variants, "suite": summary.get("suite")}, reference_variant=reference_variant),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit baseline matrix behaviour.")
    parser.add_argument("--suite-summary", type=str, required=True, help="Path to suite_summary.json")
    parser.add_argument("--reference-variant", type=str, default="single_pass_zeroshot")
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-md", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = audit_suite(Path(args.suite_summary), reference_variant=args.reference_variant)

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown = render_markdown(audit, reference_variant=args.reference_variant)
    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(markdown, encoding="utf-8")

    print(markdown)


if __name__ == "__main__":
    main()
