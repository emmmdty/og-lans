#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate whether evaluation artifacts satisfy publication-oriented fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from oglans.utils.academic_eval import API_SUITE_REPORT_METRICS, extract_report_metrics
from oglans.utils.compare_contract import COMPARABLE_CONTRACT_FIELDS


MINI_SUITE_REPORT_METRICS = (
    "doc_role_micro_f1",
    "doc_instance_micro_f1",
    "doc_combination_micro_f1",
    "doc_event_type_micro_f1",
    "single_event_doc_role_micro_f1",
    "multi_event_doc_role_micro_f1",
    "strict_f1",
    "relaxed_f1",
    "type_f1",
    "schema_compliance_rate",
    "hallucination_rate",
)
LOCAL_SUITE_REPORT_METRICS = MINI_SUITE_REPORT_METRICS + (
    "strict_precision",
    "strict_recall",
)


def get_nested(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def validate_eval_summary(summary: Dict[str, Any]) -> List[str]:
    required_paths = [
        "meta.timestamp",
        "meta.model",
        "meta.api_response_models",
        "meta.seed",
        "meta.command",
        "meta.config_hash_sha256",
        "meta.protocol_path",
        "meta.protocol_hash_sha256",
        "meta.eval_protocol_path",
        "meta.eval_protocol_hash",
        "meta.role_alias_path",
        "meta.role_alias_hash",
        "meta.prompt_builder_version",
        "meta.parser_version",
        "meta.normalization_version",
        "meta.training_mode",
        "meta.primary_metric",
        "meta.canonical_metric_mode",
        "meta.generation.temperature",
        "meta.generation.max_tokens",
        "meta.prompt_hashes",
        "token_usage.total_tokens",
        "token_usage.avg_tokens_per_sample",
        "api_stats.failed_calls",
        "runtime.wall_clock_seconds",
        "runtime_manifest.python.version",
        "runtime_manifest.system.platform",
    ]
    errors: List[str] = []
    for path in required_paths:
        value = get_nested(summary, path)
        if value is None:
            errors.append(f"Missing required field: {path}")

    compare = summary.get("compare")
    if not isinstance(compare, dict):
        errors.append("Missing compare block")
    else:
        for field_name in COMPARABLE_CONTRACT_FIELDS:
            if compare.get(field_name) is None:
                errors.append(f"Missing required field: compare.{field_name}")

    diagnostics = summary.get("diagnostics")
    if not isinstance(diagnostics, dict):
        errors.append("Missing diagnostics block")

    cost = summary.get("cost")
    if not isinstance(cost, dict):
        errors.append("Missing cost block")

    has_gold = bool(get_nested(summary, "meta.has_gold_labels"))
    metrics = get_nested(summary, "metrics")
    if not isinstance(metrics, dict):
        errors.append("metrics must be a dict")
        return errors

    if has_gold:
        for metric_name in [
            "doc_role_micro_f1",
            "doc_instance_micro_f1",
            "doc_combination_micro_f1",
            "doc_event_type_micro_f1",
            "single_event_doc_role_micro_f1",
            "multi_event_doc_role_micro_f1",
            "strict_f1",
            "relaxed_f1",
            "type_f1",
            "parse_error_rate",
            "parse_success_rate",
            "primary_metric",
            "primary_metric_value",
        ]:
            if metric_name not in metrics:
                errors.append(f"Missing metric: {metric_name}")
        if "academic_metrics" not in metrics:
            errors.append("Missing metric: academic_metrics")
        if "bootstrap_ci" not in metrics:
            errors.append("Missing metric: bootstrap_ci (enable --compute_ci)")
    else:
        for metric_name in ["evaluation_mode", "parse_error_rate", "parse_success_rate"]:
            if metric_name not in metrics:
                errors.append(f"Missing prediction-only metric: {metric_name}")

    analysis = get_nested(summary, "analysis")
    if not isinstance(analysis, dict):
        errors.append("Missing analysis block")
    else:
        if analysis.get("primary_metric") is None:
            errors.append("Missing analysis.primary_metric")
        if analysis.get("protocol") is None:
            errors.append("Missing analysis.protocol")

    return errors


def _suite_kind(summary: Mapping[str, Any]) -> str | None:
    if isinstance(summary.get("records"), list):
        return "mini"
    if isinstance(summary.get("runs"), list) and "shared_reference_meta" in summary:
        return "local"
    if isinstance(summary.get("runs"), list) and "modes" in summary:
        return "api"
    return None


def _suite_expected_metrics(kind: str) -> Tuple[str, ...]:
    if kind == "mini":
        return MINI_SUITE_REPORT_METRICS
    if kind == "local":
        return LOCAL_SUITE_REPORT_METRICS
    if kind == "api":
        return API_SUITE_REPORT_METRICS
    raise ValueError(f"Unknown suite kind: {kind}")


def _suite_group_key(kind: str, record: Mapping[str, Any]) -> str | None:
    if kind == "mini":
        run_key = record.get("run_key")
        prompt_variant = record.get("prompt_variant")
        if not run_key or not prompt_variant:
            return None
        return f"{run_key}/{prompt_variant}"
    if kind == "local":
        run_key = record.get("run_key")
        return str(run_key) if run_key else None
    if kind == "api":
        mode = record.get("mode")
        return str(mode) if mode else None
    return None


def _iter_suite_records(summary: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    if isinstance(summary.get("records"), list):
        yield from summary["records"]  # type: ignore[index]
    elif isinstance(summary.get("runs"), list):
        yield from summary["runs"]  # type: ignore[index]


def _load_suite_child_summaries(
    summary: Mapping[str, Any],
    *,
    kind: str,
    expected_metrics: Tuple[str, ...],
) -> Tuple[Dict[str, List[Dict[str, float]]], List[str]]:
    grouped: Dict[str, List[Dict[str, float]]] = {}
    errors: List[str] = []
    for record in _iter_suite_records(summary):
        summary_file = record.get("summary_file")
        if not summary_file:
            continue
        group_key = _suite_group_key(kind, record)
        if group_key is None:
            continue
        summary_path = Path(str(summary_file))
        if not summary_path.exists():
            errors.append(f"Missing child summary file: {summary_path}")
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            child_summary = json.load(handle)
        try:
            grouped.setdefault(group_key, []).append(
                extract_report_metrics(child_summary, required_metrics=expected_metrics)
            )
        except ValueError as exc:
            errors.append(f"{summary_path}: {exc}")
    return grouped, errors


def _validate_suite_aggregates(
    summary: Mapping[str, Any],
    *,
    kind: str,
    expected_metrics: Tuple[str, ...],
) -> List[str]:
    errors: List[str] = []
    aggregated = summary.get("aggregated")
    if not isinstance(aggregated, Mapping):
        return ["aggregated must be a dict"]

    child_metrics_by_group, child_errors = _load_suite_child_summaries(
        summary,
        kind=kind,
        expected_metrics=expected_metrics,
    )
    errors.extend(child_errors)

    normalized: Dict[str, Mapping[str, Any]] = {}
    if kind == "mini":
        for run_key, prompt_map in aggregated.items():
            if not isinstance(prompt_map, Mapping):
                errors.append(f"aggregated[{run_key}] must be a dict")
                continue
            for prompt_variant, payload in prompt_map.items():
                normalized[f"{run_key}/{prompt_variant}"] = payload
    else:
        normalized = {str(group_key): payload for group_key, payload in aggregated.items() if isinstance(payload, Mapping)}

    for group_key, payload in normalized.items():
        metrics_block = payload.get("metrics")
        if not isinstance(metrics_block, Mapping):
            errors.append(f"aggregated[{group_key}].metrics must be a dict")
            continue
        if group_key not in child_metrics_by_group:
            errors.append(f"Missing child summaries for aggregated group: {group_key}")
            continue
        child_rows = child_metrics_by_group[group_key]
        for metric_name in expected_metrics:
            metric_payload = metrics_block.get(metric_name)
            if not isinstance(metric_payload, Mapping):
                errors.append(f"Missing aggregated metric: {group_key}.{metric_name}")
                continue
            mean_value = metric_payload.get("mean")
            if mean_value is None:
                errors.append(f"Missing aggregated mean: {group_key}.{metric_name}")
                continue
            expected_mean = sum(row[metric_name] for row in child_rows) / len(child_rows)
            if abs(float(mean_value) - expected_mean) > 1e-9:
                errors.append(
                    f"Aggregated mean mismatch for {group_key}.{metric_name}: "
                    f"expected {expected_mean:.6f}, got {float(mean_value):.6f}"
                )
    return errors


def validate_suite_summary(summary: Dict[str, Any]) -> List[str]:
    kind = _suite_kind(summary)
    if kind is None:
        return ["Unsupported suite summary structure"]

    errors: List[str] = []
    for field_name in ("timestamp", "config", "dataset", "split", "primary_metric", "aggregated", "significance"):
        if summary.get(field_name) is None:
            errors.append(f"Missing suite field: {field_name}")

    seeds = summary.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        errors.append("Missing suite field: seeds")

    if isinstance(seeds, list) and len(seeds) < 2:
        for field_name in ("significance_status", "significance_min_pairs", "significance_skipped_reason"):
            if summary.get(field_name) is None:
                errors.append(f"Missing suite field: {field_name}")
        if summary.get("significance") not in ({}, None):
            errors.append("Single-seed suite must not report significance statistics")

    expected_metrics = _suite_expected_metrics(kind)
    errors.extend(_validate_suite_aggregates(summary, kind=kind, expected_metrics=expected_metrics))
    return errors


def validate_summary(summary: Dict[str, Any]) -> List[str]:
    if "meta" in summary and "metrics" in summary:
        return validate_eval_summary(summary)
    return validate_suite_summary(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate academic evaluation artifact.")
    parser.add_argument("--summary", type=str, required=True, help="Path to summary JSON file")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    errors = validate_summary(summary)
    if errors:
        print("[FAIL] Artifact does not satisfy required academic fields:")
        for error in errors:
            print(f" - {error}")
        raise SystemExit(1)
    print("[PASS] Artifact includes required academic reporting fields.")


if __name__ == "__main__":
    main()
