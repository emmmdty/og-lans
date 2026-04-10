"""
Shared compare-contract and diagnostics helpers for cross-family evaluation.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence


COMPARABLE_CONTRACT_FIELDS = (
    "model_family",
    "model_kind",
    "split",
    "primary_metric",
    "stage_mode",
    "prompt_variant",
    "fewshot_num_examples",
    "fewshot_selection_mode",
    "fewshot_pool_split",
    "train_tune_ratio",
    "research_split_manifest_path",
    "research_split_manifest_hash",
    "pipeline_mode",
    "canonical_metric_mode",
    "protocol_hash",
    "role_alias_hash",
    "seed",
    "seed_effective",
    "token_usage_kind",
    "comparable_contract_hash",
)

COMPARABLE_HASH_FIELDS = (
    "split",
    "primary_metric",
    "stage_mode",
    "prompt_variant",
    "fewshot_num_examples",
    "fewshot_selection_mode",
    "fewshot_pool_split",
    "train_tune_ratio",
    "research_split_manifest_hash",
    "pipeline_mode",
    "canonical_metric_mode",
    "protocol_hash",
    "role_alias_hash",
)


def _stable_hash(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_compare_contract(payload: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = {key: payload.get(key) for key in COMPARABLE_CONTRACT_FIELDS if key != "comparable_contract_hash"}
    missing = [key for key, value in normalized.items() if value is None]
    if missing:
        raise ValueError(f"Missing compare-contract fields: {', '.join(sorted(missing))}")
    normalized["comparable_contract_hash"] = _stable_hash(
        {key: normalized[key] for key in COMPARABLE_HASH_FIELDS}
    )
    return normalized


def extract_compare_contract(payload: Mapping[str, Any]) -> Dict[str, Any]:
    compare = payload.get("compare")
    if not isinstance(compare, Mapping):
        raise ValueError("Missing compare block")
    return build_compare_contract(compare)


def validate_compare_contract_match(compare_blocks: Sequence[Mapping[str, Any]]) -> str:
    hashes = {
        str(build_compare_contract(compare)["comparable_contract_hash"])
        for compare in compare_blocks
    }
    if len(hashes) != 1:
        raise ValueError("comparable contract mismatch across evaluation records")
    return next(iter(hashes))


def _event_types(events: Iterable[Mapping[str, Any]]) -> List[str]:
    seen: List[str] = []
    for event in events:
        event_type = str(event.get("event_type", "")).strip()
        if event_type and event_type not in seen:
            seen.append(event_type)
    return seen


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def build_result_diagnostics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    parse_success = 0
    total_gold_events = 0
    total_pred_events = 0
    total_gold_types = 0
    total_schema_size = 0
    total_stage1_predicted_types = 0
    stage1_rows = 0
    stage1_gold_covered = 0
    stage1_exact = 0
    stage1_missed = 0
    stage1_overpredicted = 0
    fewshot_example_counter: Counter[str] = Counter()
    fewshot_combo_counter: Counter[str] = Counter()
    correction_rows = 0
    correction_applied = 0
    records_split_count = 0
    roles_rewritten_count = 0
    roles_added_count = 0
    events_dropped_after_correction = 0
    correction_trigger_counter: Counter[str] = Counter()

    for row in rows:
        if bool(row.get("parse_success")):
            parse_success += 1

        gold_events = row.get("ground_truth") or row.get("gold") or []
        pred_events = (
            row.get("prediction_canonical")
            or row.get("pred_canonical")
            or row.get("prediction")
            or row.get("pred")
            or []
        )
        total_gold_events += len(gold_events)
        total_pred_events += len(pred_events)
        gold_types = set(_event_types(gold_events))
        total_gold_types += len(gold_types)

        prompt_meta = row.get("prompt_meta") or {}
        example_ids = [str(item) for item in prompt_meta.get("fewshot_example_ids", []) if str(item)]
        for example_id in example_ids:
            fewshot_example_counter[example_id] += 1
        if example_ids:
            fewshot_combo_counter[" | ".join(example_ids)] += 1

        correction_stats = row.get("correction_stats") or {}
        if isinstance(correction_stats, Mapping):
            correction_rows += 1
            if bool(correction_stats.get("applied")):
                correction_applied += 1
            records_split_count += int(correction_stats.get("records_split_count", 0) or 0)
            roles_rewritten_count += int(correction_stats.get("roles_rewritten_count", 0) or 0)
            roles_added_count += int(correction_stats.get("roles_added_count", 0) or 0)
            events_dropped_after_correction += int(
                correction_stats.get("events_dropped_after_correction", 0) or 0
            )
            for key, value in (correction_stats.get("correction_trigger_breakdown") or {}).items():
                correction_trigger_counter[str(key)] += int(value or 0)

        stage_meta = row.get("stage_meta") or {}
        schema_types = [str(item) for item in stage_meta.get("stage2_schema_event_types", []) if str(item)]
        total_schema_size += len(schema_types)

        if str(stage_meta.get("stage_mode", "single_pass")) != "two_stage":
            continue

        stage1_rows += 1
        predicted_types = {
            str(item).strip()
            for item in stage_meta.get("stage1_predicted_event_types", [])
            if str(item).strip()
        }
        total_stage1_predicted_types += len(predicted_types)
        if gold_types and gold_types.issubset(predicted_types):
            stage1_gold_covered += 1
        elif gold_types:
            stage1_missed += 1
        if predicted_types == gold_types:
            stage1_exact += 1
        if predicted_types - gold_types:
            stage1_overpredicted += 1

    return {
        "parse_success_rate": _safe_div(parse_success, total),
        "parse_error_rate": _safe_div(total - parse_success, total),
        "avg_gold_events": _safe_div(total_gold_events, total),
        "avg_predicted_events": _safe_div(total_pred_events, total),
        "avg_gold_event_types": _safe_div(total_gold_types, total),
        "avg_schema_event_types": _safe_div(total_schema_size, total),
        "stage1_gold_coverage_rate": _safe_div(stage1_gold_covered, stage1_rows) if stage1_rows else None,
        "stage1_exact_match_rate": _safe_div(stage1_exact, stage1_rows) if stage1_rows else None,
        "stage1_miss_rate": _safe_div(stage1_missed, stage1_rows) if stage1_rows else None,
        "stage1_overprediction_rate": _safe_div(stage1_overpredicted, stage1_rows) if stage1_rows else None,
        "avg_stage1_predicted_types": _safe_div(total_stage1_predicted_types, stage1_rows) if stage1_rows else None,
        "fewshot_unique_example_ids": len(fewshot_example_counter),
        "fewshot_unique_combinations": len(fewshot_combo_counter),
        "fewshot_top_examples": [
            {"example_id": example_id, "count": count}
            for example_id, count in fewshot_example_counter.most_common(10)
        ],
        "fewshot_top_combinations": [
            {"example_ids": combo.split(" | "), "count": count}
            for combo, count in fewshot_combo_counter.most_common(10)
        ],
        "correction_applied_rate": _safe_div(correction_applied, correction_rows) if correction_rows else None,
        "records_split_count": records_split_count,
        "roles_rewritten_count": roles_rewritten_count,
        "roles_added_count": roles_added_count,
        "events_dropped_after_correction": events_dropped_after_correction,
        "correction_trigger_breakdown": dict(sorted(correction_trigger_counter.items())),
    }
