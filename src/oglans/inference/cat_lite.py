"""
CAT-lite inference helpers.

This module provides a lightweight "choose-after-think" post-processing stage
that can be plugged into existing e2e extraction results without changing the
training pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or "")).lower()


def _build_role_schema(schema: Optional[Dict[str, Sequence[str]]]) -> Dict[str, Set[str]]:
    role_schema: Dict[str, Set[str]] = {}
    if not isinstance(schema, dict):
        return role_schema
    for event_type, roles in schema.items():
        if not isinstance(event_type, str):
            continue
        if isinstance(roles, (list, tuple, set)):
            role_schema[event_type] = {
                str(role)
                for role in roles
                if isinstance(role, str) and str(role).strip()
            }
        else:
            role_schema[event_type] = set()
    return role_schema


@dataclass
class CatLiteResult:
    events: List[Dict[str, Any]]
    kept_events: int
    dropped_events: int
    kept_arguments: int
    dropped_arguments: int


def apply_cat_lite_pipeline(
    pred_events: Optional[List[Dict[str, Any]]],
    source_text: str,
    schema: Optional[Dict[str, Sequence[str]]] = None,
    require_argument_in_text: bool = True,
) -> CatLiteResult:
    """
    Filter predicted events with lightweight schema/text evidence constraints.
    """
    events = pred_events if isinstance(pred_events, list) else []
    clean_source = _normalize_text(source_text)
    role_schema = _build_role_schema(schema)
    valid_event_types = set(role_schema.keys()) if role_schema else set()

    kept_events = 0
    dropped_events = 0
    kept_arguments = 0
    dropped_arguments = 0
    normalized_events: List[Dict[str, Any]] = []

    for event in events:
        if not isinstance(event, dict):
            dropped_events += 1
            continue

        event_type = str(event.get("event_type", "")).strip()
        if valid_event_types and event_type not in valid_event_types:
            dropped_events += 1
            continue

        allowed_roles = role_schema.get(event_type, set())
        args = event.get("arguments", [])
        if not isinstance(args, list):
            dropped_events += 1
            continue

        kept_args: List[Dict[str, str]] = []
        for arg in args:
            if not isinstance(arg, dict):
                dropped_arguments += 1
                continue
            role = str(arg.get("role", "")).strip()
            value = str(arg.get("argument", "")).strip()
            if not role or not value:
                dropped_arguments += 1
                continue
            if allowed_roles and role not in allowed_roles:
                dropped_arguments += 1
                continue
            if require_argument_in_text and _normalize_text(value) not in clean_source:
                dropped_arguments += 1
                continue
            kept_args.append({"role": role, "argument": value})
            kept_arguments += 1

        if kept_args:
            normalized_events.append(
                {
                    "event_type": event_type,
                    "trigger": event.get("trigger", ""),
                    "arguments": kept_args,
                }
            )
            kept_events += 1
        else:
            dropped_events += 1

    return CatLiteResult(
        events=normalized_events,
        kept_events=kept_events,
        dropped_events=dropped_events,
        kept_arguments=kept_arguments,
        dropped_arguments=dropped_arguments,
    )


def perturb_text_for_counterfactual(
    text: str,
    target_types: Sequence[str] = ("number", "date", "org"),
) -> Tuple[str, Dict[str, Any]]:
    """
    Single-slot perturbation for counterfactual faithfulness checks.
    """
    src = str(text or "")
    requested = {str(t).lower() for t in target_types}

    patterns: List[Tuple[str, str, str]] = []
    if "number" in requested:
        patterns.extend(
            [
                (r"\d+(?:\.\d+)?(?:亿|万|千|百)?(?:元|股|%)", "9999", "number"),
                (r"\d+(?:\.\d+)?", "9999", "number"),
            ]
        )
    if "date" in requested:
        patterns.extend(
            [
                (r"\d{4}年\d{1,2}月\d{1,2}日", "2099年12月31日", "date"),
                (r"\d{4}年\d{1,2}月", "2099年12月", "date"),
                (r"\d{4}年", "2099年", "date"),
            ]
        )
    if "org" in requested:
        patterns.extend(
            [
                (r"[\u4e00-\u9fa5A-Za-z0-9]{2,30}(?:公司|集团|银行|证券|科技|股份)", "示例科技公司", "org"),
            ]
        )

    for pattern, replacement, ptype in patterns:
        match = re.search(pattern, src)
        if not match:
            continue
        old_value = match.group(0)
        new_value = replacement
        # Keep replacement meaningful for numeric styles.
        if ptype == "number" and ("%" in old_value):
            new_value = "99%"
        elif ptype == "number" and ("股" in old_value):
            new_value = "9999股"
        elif ptype == "number" and ("元" in old_value):
            new_value = "9999元"
        perturbed = src[: match.start()] + new_value + src[match.end() :]
        return perturbed, {
            "changed": True,
            "type": ptype,
            "old_value": old_value,
            "new_value": new_value,
        }

    return src, {"changed": False}

