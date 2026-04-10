"""
Deterministic record-level correction helpers shared by local and API evaluation.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from oglans.inference.cat_lite import CatLiteResult, apply_cat_lite_pipeline
from oglans.utils.json_parser import (
    _build_argument_payload,
    _build_role_schema,
    _ground_argument,
    _safe_split_multi_value_argument,
)


SUPPORTED_PIPELINE_MODES = ("e2e", "cat_lite", "record_corrector", "record_corrector+cat_lite")
_DEFAULT_GROUNDING_MODE = "exact+fuzzy+code_local"
_SPLIT_PRIMARY_ROLE_BY_EVENT = {
    "被约谈": "公司名称",
    "股东增持": "增持方",
    "股东减持": "减持方",
    "质押": "质押方",
    "解除质押": "质押方",
}
_ROLE_COMPLETION_PATTERNS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "股份回购": {
        "回购股份数量": (
            r"(?:累计)?回购(?:股份)?数量(?:为|达|约为|约)?(?P<value>\d[\d,\.]*(?:亿|万|千|百)?股)",
            r"累计回购(?P<value>\d[\d,\.]*(?:亿|万|千|百)?股)",
        ),
        "每股交易价格": (
            r"(?:每股(?:交易)?价格|回购价格)(?:不超过|不低于|为|约为|约)?(?P<value>\d[\d,\.]*(?:元/股|元))",
            r"价格(?:不超过|不低于|为|约为|约)?(?P<value>\d[\d,\.]*(?:元/股|元))",
        ),
    },
    "企业融资": {
        "融资金额": (
            r"融资(?:金额)?(?:为|达|约为|约)?(?P<value>\d[\d,\.]*(?:亿|万|千|百)?(?:元|美元|人民币))",
        ),
    },
    "质押": {
        "质押股票/股份数量": (
            r"质押(?:股票|股份)?数量(?:为|达|约为|约)?(?P<value>\d[\d,\.]*(?:亿|万|千|百)?股)",
        ),
    },
    "解除质押": {
        "质押股票/股份数量": (
            r"解除质押(?:股票|股份)?数量(?:为|达|约为|约)?(?P<value>\d[\d,\.]*(?:亿|万|千|百)?股)",
        ),
    },
}


@dataclass
class RecordCorrectionResult:
    events: List[Dict[str, Any]]
    records_split_count: int = 0
    roles_rewritten_count: int = 0
    roles_added_count: int = 0
    roles_removed_count: int = 0
    events_dropped_after_correction: int = 0
    applied: bool = False
    trigger_breakdown: Dict[str, int] = field(default_factory=dict)


@dataclass
class StructuredPipelineResult:
    events: List[Dict[str, Any]]
    correction_result: Optional[RecordCorrectionResult]
    cat_result: Optional[CatLiteResult]


def validate_pipeline_mode(pipeline_mode: Optional[str]) -> str:
    normalized = str(pipeline_mode or "e2e").strip().lower()
    if normalized not in SUPPORTED_PIPELINE_MODES:
        raise ValueError(
            f"Unsupported pipeline_mode: {normalized}. "
            f"Expected one of {', '.join(SUPPORTED_PIPELINE_MODES)}."
        )
    return normalized


def _copy_event(event_type: str, trigger: Optional[str], arguments: List[Dict[str, str]]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "event_type": event_type,
        "arguments": arguments,
    }
    if isinstance(trigger, str) and trigger.strip():
        payload["trigger"] = trigger.strip()
    return payload


def _dedupe_arguments(arguments: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    deduped: List[Dict[str, str]] = []
    for arg in arguments:
        role = str(arg.get("role", "")).strip()
        value = str(arg.get("argument", "")).strip()
        key = (role, value)
        if not role or not value or key in seen:
            continue
        seen.add(key)
        deduped.append({"role": role, "argument": value})
    return deduped


def _extract_role_candidates(event_type: str, role: str, source_text: str) -> List[str]:
    patterns = _ROLE_COMPLETION_PATTERNS.get(event_type, {}).get(role, ())
    candidates: List[str] = []
    seen = set()
    for pattern in patterns:
        for match in re.finditer(pattern, str(source_text or "")):
            value = str(match.groupdict().get("value") or "").strip(" ：:，,。；;")
            if not value or value in seen:
                continue
            grounding_status, _ = _ground_argument(value, source_text, _DEFAULT_GROUNDING_MODE)
            if grounding_status == "ungrounded":
                continue
            seen.add(value)
            candidates.append(value)
    return candidates


def _split_event_records(
    event_type: str,
    trigger: Optional[str],
    arguments: List[Dict[str, str]],
) -> Tuple[List[Dict[str, Any]], int]:
    primary_role = _SPLIT_PRIMARY_ROLE_BY_EVENT.get(event_type)
    if not primary_role:
        return [_copy_event(event_type, trigger, arguments)], 0

    primary_values = [arg["argument"] for arg in arguments if arg.get("role") == primary_role]
    primary_values = list(dict.fromkeys(primary_values))
    if len(primary_values) <= 1:
        return [_copy_event(event_type, trigger, arguments)], 0

    shared_arguments = [arg for arg in arguments if arg.get("role") != primary_role]
    split_events = [
        _copy_event(
            event_type,
            trigger,
            _dedupe_arguments([_build_argument_payload(primary_role, value)] + shared_arguments),
        )
        for value in primary_values
    ]
    return split_events, max(0, len(split_events) - 1)


def apply_record_corrector(
    pred_events: Optional[List[Dict[str, Any]]],
    *,
    source_text: str,
    schema: Optional[Dict[str, Sequence[str]]] = None,
    role_alias_map: Optional[Dict[str, Dict[str, str]]] = None,
) -> RecordCorrectionResult:
    events = pred_events if isinstance(pred_events, list) else []
    role_schema = _build_role_schema(schema)
    valid_event_types = set(role_schema.keys()) if role_schema else set()
    trigger_counter: Counter[str] = Counter()
    corrected_events: List[Dict[str, Any]] = []
    records_split_count = 0
    roles_rewritten_count = 0
    roles_added_count = 0
    roles_removed_count = 0
    events_dropped = 0

    for event in events:
        if not isinstance(event, Mapping):
            events_dropped += 1
            trigger_counter["drop:invalid_event"] += 1
            continue

        event_type = str(event.get("event_type", "")).strip()
        if valid_event_types and event_type not in valid_event_types:
            events_dropped += 1
            trigger_counter[f"drop:invalid_event_type:{event_type or 'unknown'}"] += 1
            continue

        allowed_roles = role_schema.get(event_type, set())
        alias_map = role_alias_map.get(event_type, {}) if isinstance(role_alias_map, dict) else {}
        raw_arguments = event.get("arguments", [])
        if not isinstance(raw_arguments, list):
            events_dropped += 1
            trigger_counter[f"drop:invalid_arguments:{event_type or 'unknown'}"] += 1
            continue

        normalized_args: List[Dict[str, str]] = []
        seen_argument_keys = set()
        for arg in raw_arguments:
            if not isinstance(arg, Mapping):
                roles_removed_count += 1
                trigger_counter["drop:invalid_argument"] += 1
                continue
            role = str(arg.get("role", "")).strip()
            value = str(arg.get("argument", "")).strip()
            if not role or not value:
                roles_removed_count += 1
                trigger_counter["drop:empty_role_or_argument"] += 1
                continue

            canonical_role = alias_map.get(role, role)
            if canonical_role != role:
                roles_rewritten_count += 1
                trigger_counter[f"rewrite:{event_type}:{role}->{canonical_role}"] += 1
                role = canonical_role

            if allowed_roles and role not in allowed_roles:
                roles_removed_count += 1
                trigger_counter[f"drop:invalid_role:{event_type}:{role}"] += 1
                continue

            split_values, split_result = _safe_split_multi_value_argument(
                event_type=event_type,
                role=role,
                argument=value,
                source_text=source_text,
                grounding_mode=_DEFAULT_GROUNDING_MODE,
            )
            if split_result.get("applied"):
                trigger_counter[f"split_argument:{event_type}:{role}"] += 1
            for candidate in split_values:
                key = (role, candidate)
                if key in seen_argument_keys:
                    continue
                seen_argument_keys.add(key)
                normalized_args.append(_build_argument_payload(role, candidate))

        if allowed_roles:
            present_roles = {str(arg.get("role", "")).strip() for arg in normalized_args}
            for role in sorted(allowed_roles):
                if role in present_roles:
                    continue
                for candidate in _extract_role_candidates(event_type, role, source_text):
                    key = (role, candidate)
                    if key in seen_argument_keys:
                        continue
                    normalized_args.append(_build_argument_payload(role, candidate))
                    seen_argument_keys.add(key)
                    roles_added_count += 1
                    present_roles.add(role)
                    trigger_counter[f"add:{event_type}:{role}"] += 1
                    break

        normalized_args = _dedupe_arguments(normalized_args)
        if not normalized_args:
            events_dropped += 1
            trigger_counter[f"drop:no_arguments_after_correction:{event_type or 'unknown'}"] += 1
            continue

        split_events, split_count = _split_event_records(
            event_type,
            event.get("trigger"),
            normalized_args,
        )
        if split_count:
            trigger_counter[f"split_record:{event_type}"] += split_count
            records_split_count += split_count
        corrected_events.extend(split_events)

    applied = any(
        (
            records_split_count,
            roles_rewritten_count,
            roles_added_count,
            roles_removed_count,
            events_dropped,
        )
    )
    return RecordCorrectionResult(
        events=corrected_events,
        records_split_count=records_split_count,
        roles_rewritten_count=roles_rewritten_count,
        roles_added_count=roles_added_count,
        roles_removed_count=roles_removed_count,
        events_dropped_after_correction=events_dropped,
        applied=applied,
        trigger_breakdown=dict(sorted(trigger_counter.items())),
    )


def apply_structured_event_pipeline(
    pred_events: Optional[List[Dict[str, Any]]],
    *,
    source_text: str,
    schema: Optional[Dict[str, Sequence[str]]] = None,
    role_alias_map: Optional[Dict[str, Dict[str, str]]] = None,
    pipeline_mode: str = "e2e",
) -> StructuredPipelineResult:
    normalized_pipeline_mode = validate_pipeline_mode(pipeline_mode)
    events = pred_events if isinstance(pred_events, list) else []
    correction_result: Optional[RecordCorrectionResult] = None
    cat_result: Optional[CatLiteResult] = None

    if "record_corrector" in normalized_pipeline_mode:
        correction_result = apply_record_corrector(
            events,
            source_text=source_text,
            schema=schema,
            role_alias_map=role_alias_map,
        )
        events = correction_result.events

    if "cat_lite" in normalized_pipeline_mode:
        cat_result = apply_cat_lite_pipeline(
            pred_events=events,
            source_text=source_text,
            schema=schema,
            require_argument_in_text=True,
        )
        events = cat_result.events

    return StructuredPipelineResult(
        events=events,
        correction_result=correction_result,
        cat_result=cat_result,
    )
