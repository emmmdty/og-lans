#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teacher-silver construction helpers for hybrid DuEE-Fin experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from oglans.data.adapter import DUEE_FIN_EESample
from oglans.data.prompt_builder import ChinesePromptBuilder, build_training_response
from oglans.utils.research_protocol import extract_event_types_from_events


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            record = json.loads(payload)
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record must be an object: path={path}, line={line_no}")
            records.append(record)
    return records


def save_jsonl(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
    return output_path


def load_duee_fin_text_index(dataset_file: str | Path) -> Dict[str, str]:
    text_index: Dict[str, str] = {}
    with Path(dataset_file).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            payload = line.strip()
            if not payload:
                continue
            record = json.loads(payload)
            if not isinstance(record, dict):
                raise ValueError(
                    f"DuEE-Fin dataset record must be an object: path={dataset_file}, line={line_no}"
                )
            sample_id = str(record.get("id", "")).strip()
            text = str(record.get("text", "")).strip()
            if sample_id and text:
                text_index[sample_id] = text
    return text_index


def _extract_predicted_events(row: Mapping[str, Any]) -> List[Dict[str, Any]]:
    for key in ("prediction_canonical", "pred_canonical", "prediction", "pred"):
        events = row.get(key)
        if isinstance(events, list):
            return [dict(item) for item in events if isinstance(item, dict)]
    return []


def _resolve_source_text(
    row: Mapping[str, Any],
    *,
    text_index: Optional[Mapping[str, str]] = None,
    allow_preview_text: bool = False,
) -> Optional[str]:
    sample_id = str(row.get("id", "")).strip()
    if text_index and sample_id in text_index:
        text = str(text_index[sample_id]).strip()
        if text:
            return text

    direct_text = str(row.get("text", "")).strip()
    if direct_text:
        return direct_text

    preview_text = str(row.get("text_preview", "")).strip()
    if preview_text and allow_preview_text and "..." not in preview_text:
        return preview_text
    return None


def _build_role_tuple_set(events: Sequence[Mapping[str, Any]]) -> set[Tuple[str, str, str]]:
    tuples: set[Tuple[str, str, str]] = set()
    for event in events:
        event_type = str(event.get("event_type", "")).strip()
        if not event_type:
            continue
        for argument in event.get("arguments", []) or []:
            if not isinstance(argument, Mapping):
                continue
            role = str(argument.get("role", "")).strip()
            value = str(argument.get("argument", "")).strip()
            if role and value:
                tuples.add((event_type, role, value))
    return tuples


def compute_role_overlap(
    primary_events: Sequence[Mapping[str, Any]],
    secondary_events: Sequence[Mapping[str, Any]],
) -> float:
    primary_roles = _build_role_tuple_set(primary_events)
    secondary_roles = _build_role_tuple_set(secondary_events)
    if not primary_roles and not secondary_roles:
        return 1.0
    if not primary_roles or not secondary_roles:
        return 0.0
    intersection = len(primary_roles & secondary_roles)
    return (2.0 * intersection) / (len(primary_roles) + len(secondary_roles))


def build_teacher_silver_records(
    primary_rows: Sequence[Mapping[str, Any]],
    *,
    secondary_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    text_index: Optional[Mapping[str, str]] = None,
    allowed_ids: Optional[Iterable[str]] = None,
    require_consensus: bool = False,
    require_matching_event_types: bool = True,
    parse_success_only: bool = True,
    min_role_overlap: float = 0.5,
    allow_preview_text: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    allowed_id_set = {str(item) for item in (allowed_ids or []) if str(item).strip()} or None
    secondary_index = {
        str(row.get("id", "")).strip(): row
        for row in (secondary_rows or [])
        if str(row.get("id", "")).strip()
    }

    records: List[Dict[str, Any]] = []
    skipped: Dict[str, int] = {}
    agreement_scores: List[float] = []

    def _skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for primary_row in primary_rows:
        sample_id = str(primary_row.get("id", "")).strip()
        if not sample_id:
            _skip("missing_id")
            continue
        if allowed_id_set is not None and sample_id not in allowed_id_set:
            _skip("outside_allowed_ids")
            continue
        if parse_success_only and not bool(primary_row.get("parse_success", False)):
            _skip("primary_parse_failed")
            continue

        primary_events = _extract_predicted_events(primary_row)
        if not primary_events:
            _skip("primary_empty_prediction")
            continue

        source_text = _resolve_source_text(
            primary_row,
            text_index=text_index,
            allow_preview_text=allow_preview_text,
        )
        if not source_text:
            _skip("missing_source_text")
            continue

        teacher_meta: Dict[str, Any] = {
            "sample_id": sample_id,
            "primary_pipeline_mode": primary_row.get("pipeline_mode"),
            "primary_parse_success": bool(primary_row.get("parse_success", False)),
            "consensus_mode": "single_source",
            "agreement_score": None,
        }

        if secondary_rows is not None:
            secondary_row = secondary_index.get(sample_id)
            if secondary_row is None:
                _skip("missing_secondary_row")
                continue
            if parse_success_only and not bool(secondary_row.get("parse_success", False)):
                _skip("secondary_parse_failed")
                continue
            secondary_events = _extract_predicted_events(secondary_row)
            if not secondary_events:
                _skip("secondary_empty_prediction")
                continue

            primary_types = set(extract_event_types_from_events(primary_events))
            secondary_types = set(extract_event_types_from_events(secondary_events))
            if require_matching_event_types and primary_types != secondary_types:
                _skip("event_type_mismatch")
                continue

            agreement_score = compute_role_overlap(primary_events, secondary_events)
            agreement_scores.append(agreement_score)
            if require_consensus and agreement_score < float(min_role_overlap):
                _skip("insufficient_role_overlap")
                continue

            teacher_meta.update(
                {
                    "consensus_mode": "dual_source",
                    "secondary_pipeline_mode": secondary_row.get("pipeline_mode"),
                    "secondary_parse_success": bool(secondary_row.get("parse_success", False)),
                    "agreement_score": round(float(agreement_score), 6),
                }
            )

        records.append(
            {
                "id": sample_id,
                "text": source_text,
                "event_list": primary_events,
                "teacher_meta": teacher_meta,
            }
        )

    summary = {
        "input_count": len(primary_rows),
        "kept_count": len(records),
        "skipped_count": int(sum(skipped.values())),
        "skip_breakdown": skipped,
        "consensus_enabled": bool(secondary_rows is not None),
        "require_consensus": bool(require_consensus),
        "require_matching_event_types": bool(require_matching_event_types),
        "parse_success_only": bool(parse_success_only),
        "min_role_overlap": float(min_role_overlap),
        "allowed_ids_count": len(allowed_id_set or []),
        "agreement_score_mean": round(sum(agreement_scores) / len(agreement_scores), 6)
        if agreement_scores
        else None,
    }
    return records, summary


def load_teacher_silver_samples(
    silver_jsonl_path: str | Path,
    *,
    schema: Optional[Dict[str, Sequence[str]]] = None,
    max_text_length: int = 3500,
    max_samples: Optional[int] = None,
    id_prefix: str = "teacher",
) -> List[DUEE_FIN_EESample]:
    builder = ChinesePromptBuilder()
    valid_event_types = list((schema or {}).keys())
    silver_records = load_jsonl(silver_jsonl_path)
    if max_samples is not None and int(max_samples) > 0:
        silver_records = silver_records[: int(max_samples)]

    samples: List[DUEE_FIN_EESample] = []
    for record in silver_records:
        source_id = str(record.get("id", "")).strip()
        text = str(record.get("text", "")).strip()
        events = [dict(item) for item in record.get("event_list", []) if isinstance(item, dict)]
        if not source_id or not text or not events:
            continue
        sample_id = f"{id_prefix}::{source_id}"
        event_types = extract_event_types_from_events(events, valid_event_types=valid_event_types)
        prompt = builder.build_user_prompt(text, max_length=max_text_length)
        chosen = build_training_response(events)
        samples.append(
            DUEE_FIN_EESample(
                id=sample_id,
                text=text if len(text) <= max_text_length else text[:max_text_length] + "...",
                prompt=prompt,
                chosen=chosen,
                rejected="",
                event_types=event_types,
                events=events,
            )
        )
    return samples
