#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validate whether evaluation summary artifacts satisfy publication-oriented fields.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def get_nested(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def validate_summary(summary: Dict[str, Any]) -> List[str]:
    required_paths = [
        "meta.timestamp",
        "meta.model",
        "meta.api_response_models",
        "meta.seed",
        "meta.command",
        "meta.config_hash_sha256",
        "meta.protocol_path",
        "meta.protocol_hash_sha256",
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
    for p in required_paths:
        val = get_nested(summary, p)
        if val is None:
            errors.append(f"Missing required field: {p}")

    has_gold = bool(get_nested(summary, "meta.has_gold_labels"))
    metrics = get_nested(summary, "metrics")
    if not isinstance(metrics, dict):
        errors.append("metrics must be a dict")
        return errors

    if has_gold:
        for k in ["strict_f1", "relaxed_f1", "type_f1", "parse_success_rate", "primary_metric", "primary_metric_value"]:
            if k not in metrics:
                errors.append(f"Missing metric: {k}")
        if "bootstrap_ci" not in metrics:
            errors.append("Missing metric: bootstrap_ci (enable --compute_ci)")
    else:
        for k in ["evaluation_mode", "parse_success_rate"]:
            if k not in metrics:
                errors.append(f"Missing prediction-only metric: {k}")

    analysis = get_nested(summary, "analysis")
    if not isinstance(analysis, dict):
        errors.append("Missing analysis block")
    else:
        if analysis.get("primary_metric") is None:
            errors.append("Missing analysis.primary_metric")
        if analysis.get("protocol") is None:
            errors.append("Missing analysis.protocol")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate academic evaluation summary artifact.")
    parser.add_argument("--summary", type=str, required=True, help="Path to eval summary JSON file")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    errors = validate_summary(summary)
    if errors:
        print("[FAIL] Artifact does not satisfy required academic fields:")
        for e in errors:
            print(f" - {e}")
        raise SystemExit(1)
    print("[PASS] Artifact includes required academic reporting fields.")


if __name__ == "__main__":
    main()
