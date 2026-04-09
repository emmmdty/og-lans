"""
Shared research protocol helpers for baseline comparability.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from oglans.data.prompt_builder import ChinesePromptBuilder, validate_fewshot_selection_mode


SUPPORTED_STAGE_MODES = ("single_pass", "two_stage")
SUPPORTED_FEWSHOT_POOL_SPLITS = ("train", "train_fit")
DEFAULT_TRAIN_TUNE_RATIO = 0.1


def validate_stage_mode(stage_mode: Optional[str]) -> str:
    normalized = str(stage_mode or "single_pass").strip().lower()
    if normalized not in SUPPORTED_STAGE_MODES:
        raise ValueError(
            f"Unsupported stage_mode: {normalized}. "
            f"Expected one of {', '.join(SUPPORTED_STAGE_MODES)}."
        )
    return normalized


def validate_fewshot_pool_split(pool_split: Optional[str]) -> str:
    normalized = str(pool_split or "train_fit").strip().lower()
    if normalized not in SUPPORTED_FEWSHOT_POOL_SPLITS:
        raise ValueError(
            f"Unsupported fewshot_pool_split: {normalized}. "
            f"Expected one of {', '.join(SUPPORTED_FEWSHOT_POOL_SPLITS)}."
        )
    return normalized


def _load_manifest_payload(manifest_source: Any) -> Dict[str, Any]:
    if isinstance(manifest_source, (str, Path)):
        manifest_path = Path(manifest_source)
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"research split manifest must be a dict: {manifest_path}")
        payload = dict(payload)
        payload["manifest_path"] = str(manifest_path)
        return payload
    if isinstance(manifest_source, dict):
        return dict(manifest_source)
    raise TypeError("manifest_source must be a mapping or file path.")


def normalize_research_split_manifest(
    manifest_source: Any,
    *,
    pool_split: Optional[str] = None,
) -> Dict[str, Any]:
    payload = _load_manifest_payload(manifest_source)
    fit_ids = [str(item) for item in payload.get("fit_ids", []) if str(item).strip()]
    tune_ids = [str(item) for item in payload.get("tune_ids", []) if str(item).strip()]
    fit_set = set(fit_ids)
    tune_set = set(tune_ids)
    if fit_set & tune_set:
        raise ValueError("research split manifest fit_ids and tune_ids must be disjoint.")

    normalized = {
        "seed": int(payload.get("seed", 3407)),
        "tune_ratio": float(payload.get("tune_ratio", DEFAULT_TRAIN_TUNE_RATIO)),
        "fit_ids": fit_ids,
        "tune_ids": tune_ids,
        "fit_count": int(payload.get("fit_count", len(fit_ids))),
        "tune_count": int(payload.get("tune_count", len(tune_ids))),
        "pool_split": validate_fewshot_pool_split(pool_split or payload.get("pool_split", "train_fit")),
    }
    for optional_key in ("dataset", "source_split", "manifest_path"):
        if payload.get(optional_key) is not None:
            normalized[optional_key] = payload.get(optional_key)
    return normalized


def save_research_split_manifest(payload: Dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return path


def resolve_stage_settings(
    *,
    stage_mode: Optional[str] = None,
    fewshot_selection_mode: Optional[str] = None,
    fewshot_pool_split: Optional[str] = None,
    comparison_cfg: Optional[Dict[str, Any]] = None,
    default_stage_mode: str = "single_pass",
    default_selection_mode: str = "dynamic",
    default_pool_split: str = "train_fit",
) -> Dict[str, str]:
    cfg = dict(comparison_cfg or {})
    return {
        "stage_mode": validate_stage_mode(stage_mode or cfg.get("stage_mode", default_stage_mode)),
        "fewshot_selection_mode": validate_fewshot_selection_mode(
            fewshot_selection_mode or cfg.get("fewshot_selection_mode", default_selection_mode)
        ),
        "fewshot_pool_split": validate_fewshot_pool_split(
            fewshot_pool_split or cfg.get("fewshot_pool_split", default_pool_split)
        ),
    }


def _stable_sample_key(sample: Any, seed: int) -> str:
    sample_id = str(getattr(sample, "id", "") or "")
    event_types = sorted({str(item) for item in getattr(sample, "event_types", []) if item})
    signature = "|".join(event_types) if event_types else "__no_event__"
    digest = hashlib.sha256(f"{seed}:{signature}:{sample_id}".encode("utf-8")).hexdigest()
    return digest


def _sample_signature(sample: Any) -> Tuple[str, ...]:
    event_types = sorted({str(item) for item in getattr(sample, "event_types", []) if item})
    return tuple(event_types) if event_types else ("__no_event__",)


def _allocate_bucket_tune_counts(
    buckets: Dict[Tuple[str, ...], List[Any]],
    *,
    tune_ratio: float,
) -> Dict[Tuple[str, ...], int]:
    total = sum(len(items) for items in buckets.values())
    target_total = int(round(total * tune_ratio))
    allocations: Dict[Tuple[str, ...], int] = {}
    remainders: List[Tuple[float, Tuple[str, ...]]] = []
    allocated = 0

    for signature, items in buckets.items():
        raw = len(items) * tune_ratio
        base = int(raw)
        allocations[signature] = min(base, len(items))
        allocated += allocations[signature]
        remainders.append((raw - base, signature))

    for _, signature in sorted(remainders, reverse=True):
        if allocated >= target_total:
            break
        if allocations[signature] >= len(buckets[signature]):
            continue
        allocations[signature] += 1
        allocated += 1

    return allocations


def build_research_split_manifest(
    samples: Sequence[Any],
    *,
    tune_ratio: float = 0.1,
    seed: int = 3407,
) -> Dict[str, Any]:
    if not 0.0 < float(tune_ratio) < 1.0:
        raise ValueError("tune_ratio must be in (0, 1).")

    buckets: Dict[Tuple[str, ...], List[Any]] = {}
    for sample in samples:
        buckets.setdefault(_sample_signature(sample), []).append(sample)

    allocations = _allocate_bucket_tune_counts(buckets, tune_ratio=float(tune_ratio))
    fit_ids: List[str] = []
    tune_ids: List[str] = []

    for signature, items in sorted(buckets.items()):
        ordered = sorted(items, key=lambda sample: _stable_sample_key(sample, seed))
        tune_count = allocations.get(signature, 0)
        tune_bucket = ordered[:tune_count]
        fit_bucket = ordered[tune_count:]
        tune_ids.extend(str(getattr(sample, "id", "")) for sample in tune_bucket)
        fit_ids.extend(str(getattr(sample, "id", "")) for sample in fit_bucket)

    return {
        "seed": int(seed),
        "tune_ratio": float(tune_ratio),
        "fit_ids": fit_ids,
        "tune_ids": tune_ids,
        "fit_count": len(fit_ids),
        "tune_count": len(tune_ids),
    }


def split_research_samples(
    samples: Sequence[Any],
    *,
    tune_ratio: float = 0.1,
    seed: int = 3407,
) -> Tuple[List[Any], List[Any], Dict[str, Any]]:
    manifest = build_research_split_manifest(samples, tune_ratio=tune_ratio, seed=seed)
    fit_ids = set(manifest["fit_ids"])
    tune_ids = set(manifest["tune_ids"])
    fit_samples = [sample for sample in samples if str(getattr(sample, "id", "")) in fit_ids]
    tune_samples = [sample for sample in samples if str(getattr(sample, "id", "")) in tune_ids]
    return fit_samples, tune_samples, manifest


def select_fewshot_pool_samples(
    samples: Sequence[Any],
    *,
    pool_split: str = "train_fit",
    tune_ratio: float = 0.1,
    seed: int = 3407,
    split_manifest: Any = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    normalized_pool_split = validate_fewshot_pool_split(pool_split)
    if split_manifest is not None:
        manifest = normalize_research_split_manifest(split_manifest, pool_split=normalized_pool_split)
        if normalized_pool_split == "train":
            return list(samples), manifest
        fit_ids = set(manifest["fit_ids"])
        selected = [
            sample for sample in samples if str(getattr(sample, "id", "")) in fit_ids
        ]
        return selected, manifest

    if normalized_pool_split == "train":
        manifest = build_research_split_manifest(samples, tune_ratio=tune_ratio, seed=seed)
        manifest["pool_split"] = normalized_pool_split
        return list(samples), manifest

    fit_samples, _, manifest = split_research_samples(samples, tune_ratio=tune_ratio, seed=seed)
    manifest["pool_split"] = normalized_pool_split
    return fit_samples, manifest


def restrict_schema_to_event_types(
    schema: Optional[Dict[str, Sequence[str]]],
    predicted_event_types: Iterable[str],
) -> Tuple[Dict[str, List[str]], List[str]]:
    normalized_schema = {
        str(event_type): [str(role) for role in list(roles or [])]
        for event_type, roles in (schema or {}).items()
        if event_type
    }
    if not normalized_schema:
        return {}, []

    predicted = {
        str(event_type)
        for event_type in predicted_event_types
        if str(event_type).strip()
    }
    selected_event_types = [
        event_type for event_type in normalized_schema.keys() if event_type in predicted
    ]
    if not selected_event_types:
        selected_event_types = list(normalized_schema.keys())

    filtered = {event_type: normalized_schema[event_type] for event_type in selected_event_types}
    return filtered, selected_event_types


def extract_event_types_from_events(
    events: Iterable[Dict[str, Any]],
    *,
    valid_event_types: Optional[Sequence[str]] = None,
) -> List[str]:
    seen = []
    whitelist = {str(item) for item in (valid_event_types or []) if item}
    for event in events:
        if not isinstance(event, dict):
            continue
        event_type = str(event.get("event_type", "")).strip()
        if not event_type:
            continue
        if whitelist and event_type not in whitelist:
            continue
        if event_type not in seen:
            seen.append(event_type)
    return seen


def build_fewshot_example_pool(
    samples: Sequence[Any],
    *,
    schema: Optional[Dict[str, List[str]]] = None,
    source_split: str = "train",
) -> List[Dict[str, Any]]:
    pool: List[Dict[str, Any]] = []
    for sample in samples:
        events = list(getattr(sample, "events", []) or [])
        if not events:
            continue
        event_types: List[str] = []
        triggers: List[str] = []
        roles: List[str] = []
        keywords: List[str] = []
        for event in events:
            event_type = event.get("event_type")
            if isinstance(event_type, str) and event_type and event_type not in event_types:
                event_types.append(event_type)
                keywords.append(event_type)
            trigger = event.get("trigger")
            if isinstance(trigger, str) and trigger and trigger not in triggers:
                triggers.append(trigger)
                keywords.append(trigger)
            for argument in event.get("arguments", []):
                role = argument.get("role")
                if isinstance(role, str) and role and role not in roles:
                    roles.append(role)
                value = argument.get("argument")
                if isinstance(value, str) and value and len(value) <= 12 and value not in keywords:
                    keywords.append(value)
        user_prompt = ChinesePromptBuilder.build_user_prompt(str(getattr(sample, "text", "")))
        assistant_prompt = ChinesePromptBuilder.build_cot_response(events, schema=schema)
        pool.append(
            {
                "id": f"{source_split}:{getattr(sample, 'id', f'sample-{len(pool)}')}",
                "user": user_prompt,
                "assistant": assistant_prompt,
                "source_text": str(getattr(sample, "text", "")),
                "event_types": event_types or list(getattr(sample, "event_types", []) or []),
                "triggers": triggers,
                "roles": roles,
                "keywords": keywords,
            }
        )
    return pool
