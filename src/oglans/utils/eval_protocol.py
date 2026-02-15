#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation protocol and role-alias helpers.
Dependency-light utilities shared by evaluation entrypoints.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

DEFAULT_EVAL_PROTOCOL: Dict[str, Any] = {
    "version": "1.0",
    "primary_metric": "strict_f1",
    "canonical_metric_mode": "analysis_only",
    "evaluation": {
        "split": "dev",
        "seeds": [3407, 3408, 3409],
        "bootstrap_samples": 1000,
        "concurrency": 8,
        "significance": "paired_permutation",
        "confidence": 0.95,
    },
}


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_eval_protocol(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return copy.deepcopy(DEFAULT_EVAL_PROTOCOL)
    p = Path(path)
    if not p.exists():
        return copy.deepcopy(DEFAULT_EVAL_PROTOCOL)
    payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Protocol must be a dict: {path}")
    return deep_merge_dict(DEFAULT_EVAL_PROTOCOL, payload)


def compute_file_hash(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_role_alias_map(path: Optional[str]) -> Dict[str, Dict[str, str]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}

    if p.suffix.lower() == ".json":
        payload = json.loads(p.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}

    root = payload.get("event_role_aliases", payload)
    if not isinstance(root, dict):
        return {}

    normalized: Dict[str, Dict[str, str]] = {}
    for event_type, role_map in root.items():
        if not isinstance(role_map, dict):
            continue
        event_key = str(event_type)
        normalized[event_key] = {}
        for alias, canonical in role_map.items():
            if not alias or not canonical:
                continue
            normalized[event_key][str(alias)] = str(canonical)
    return normalized


def canonicalize_pred_roles(
    pred_events: List[Dict[str, Any]],
    alias_map: Dict[str, Dict[str, str]],
) -> Tuple[List[Dict[str, Any]], int]:
    if not isinstance(pred_events, list) or not alias_map:
        return pred_events if isinstance(pred_events, list) else [], 0

    rewritten = 0
    normalized_events: List[Dict[str, Any]] = []
    for event in pred_events:
        if not isinstance(event, dict):
            continue
        event_type = event.get("event_type")
        role_map = alias_map.get(str(event_type), {}) if event_type else {}
        new_event = dict(event)
        args = event.get("arguments", [])
        if isinstance(args, list):
            new_args: List[Dict[str, Any]] = []
            for arg in args:
                if not isinstance(arg, dict):
                    continue
                new_arg = dict(arg)
                role = new_arg.get("role")
                if isinstance(role, str) and role in role_map:
                    mapped = role_map[role]
                    if mapped != role:
                        rewritten += 1
                    new_arg["role"] = mapped
                new_args.append(new_arg)
            new_event["arguments"] = new_args
        normalized_events.append(new_event)
    return normalized_events, rewritten
