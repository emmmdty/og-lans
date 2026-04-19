"""
Formal post-process profiles shared by local and API evaluation entrypoints.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from oglans.utils.experiment_contract import normalize_postprocess_profile

from .event_probe import apply_event_probe_v2


def _normalize_records(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for record in records or []:
        if not isinstance(record, Mapping):
            continue
        event_type = str(record.get("event_type", "")).strip()
        arguments = []
        for item in record.get("arguments", []) or []:
            if not isinstance(item, Mapping):
                continue
            role = str(item.get("role", "")).strip()
            argument = str(item.get("argument", "")).strip()
            if role and argument:
                arguments.append({"role": role, "argument": argument})
        normalized.append(
            {
                "event_type": event_type,
                "arguments": arguments,
            }
        )
    return normalized


def apply_postprocess_profile(
    records: Iterable[Mapping[str, Any]],
    *,
    source_text: str = "",
    profile: Any = "none",
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    normalized_profile = normalize_postprocess_profile(profile)
    normalized_records = _normalize_records(records)
    result_records = deepcopy(normalized_records)
    profile_stats: Dict[str, Any] = {}

    if normalized_profile == "event_probe_v2":
        result_records, profile_stats = apply_event_probe_v2(
            normalized_records,
            source_text=source_text,
        )

    return result_records, {
        "profile": normalized_profile,
        "input_records": len(normalized_records),
        "output_records": len(result_records),
        "changed": result_records != normalized_records,
        "profile_stats": dict(profile_stats),
    }


def summarize_postprocess_profile_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total_rows = len(rows)
    changed_samples = 0
    total_input_records = 0
    total_output_records = 0
    aggregated_profile_stats: Dict[str, int] = {}

    for row in rows:
        stats = row.get("postprocess_profile_stats") or {}
        if not isinstance(stats, Mapping):
            continue
        if bool(stats.get("changed", False)):
            changed_samples += 1
        total_input_records += int(stats.get("input_records", 0) or 0)
        total_output_records += int(stats.get("output_records", 0) or 0)
        for key, value in (stats.get("profile_stats") or {}).items():
            if isinstance(value, bool):
                aggregated_profile_stats[str(key)] = aggregated_profile_stats.get(str(key), 0) + int(value)
            elif isinstance(value, int):
                aggregated_profile_stats[str(key)] = aggregated_profile_stats.get(str(key), 0) + value

    return {
        "postprocess_profile_changed_samples": changed_samples,
        "postprocess_profile_changed_rate": (
            changed_samples / total_rows if total_rows else 0.0
        ),
        "postprocess_profile_input_records": total_input_records,
        "postprocess_profile_output_records": total_output_records,
        "postprocess_profile_stats": aggregated_profile_stats,
    }


__all__ = ["apply_postprocess_profile", "summarize_postprocess_profile_rows"]
