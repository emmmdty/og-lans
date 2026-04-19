"""
Training-side protocol helpers shared by the trainer and wrapper tests.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from oglans.data.prompt_builder import ChinesePromptBuilder, build_inference_prompt_payload
from oglans.utils.experiment_contract import is_two_stage_mode
from oglans.utils.research_protocol import (
    extract_event_types_from_events,
    normalize_research_split_manifest,
    restrict_schema_to_event_types,
    split_research_samples,
    validate_stage_mode,
)


def select_training_fit_samples(
    samples: Sequence[Any],
    *,
    tune_ratio: float = 0.1,
    seed: int = 3407,
    split_manifest: Any = None,
) -> Tuple[List[Any], Dict[str, Any]]:
    if split_manifest is not None:
        manifest = normalize_research_split_manifest(split_manifest, pool_split="train_fit")
        fit_ids = set(manifest["fit_ids"])
        selected = [sample for sample in samples if str(getattr(sample, "id", "")) in fit_ids]
        return selected, manifest

    fit_samples, _, manifest = split_research_samples(
        samples,
        tune_ratio=tune_ratio,
        seed=seed,
    )
    manifest["pool_split"] = "train_fit"
    return fit_samples, manifest


def resolve_trainer_sample_sources(
    training_input_samples: Sequence[Any],
    *,
    gold_source_samples: Optional[Sequence[Any]] = None,
    tune_ratio: float = 0.1,
    seed: int = 3407,
    split_manifest: Any = None,
) -> Dict[str, Any]:
    """
    Resolve the final trainer input and the gold-only source used for split metadata.

    When ``gold_source_samples`` is provided, callers have already selected the
    final training input. The trainer must not apply train-fit filtering again,
    otherwise teacher-silver IDs such as ``teacher::<source_id>`` are dropped.
    """
    final_training_samples = list(training_input_samples)
    if gold_source_samples is None:
        selected, manifest = select_training_fit_samples(
            final_training_samples,
            tune_ratio=tune_ratio,
            seed=seed,
            split_manifest=split_manifest,
        )
        return {
            "training_samples": selected,
            "fewshot_source_samples": final_training_samples,
            "split_manifest": manifest,
            "input_samples_already_selected": False,
        }

    source_samples = list(gold_source_samples)
    _, manifest = select_training_fit_samples(
        source_samples,
        tune_ratio=tune_ratio,
        seed=seed,
        split_manifest=split_manifest,
    )
    return {
        "training_samples": final_training_samples,
        "fewshot_source_samples": source_samples,
        "split_manifest": manifest,
        "input_samples_already_selected": True,
    }


def resolve_training_resume_settings(training_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(training_cfg or {})
    legacy_resume = str(cfg.get("resume_from_checkpoint") or "").strip()
    warm_start_from_checkpoint = str(cfg.get("warm_start_from_checkpoint") or "").strip()
    resume_training_from = str(cfg.get("resume_training_from") or "").strip()

    if legacy_resume:
        raise ValueError(
            "training.resume_from_checkpoint is deprecated. "
            "Use training.resume_training_from or training.warm_start_from_checkpoint."
        )
    if warm_start_from_checkpoint and resume_training_from:
        raise ValueError(
            "training.resume_training_from and training.warm_start_from_checkpoint are mutually exclusive."
        )
    if resume_training_from:
        return {
            "mode": "resume_training",
            "checkpoint_path": resume_training_from,
            "restores_training_state": True,
        }
    if warm_start_from_checkpoint:
        return {
            "mode": "warm_start",
            "checkpoint_path": warm_start_from_checkpoint,
            "restores_training_state": False,
        }
    return {
        "mode": "fresh_start",
        "checkpoint_path": None,
        "restores_training_state": False,
    }


def _stable_choice(options: Sequence[str], key: str) -> str:
    if not options:
        raise ValueError("options must not be empty")
    ordered = sorted(str(item) for item in options)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    index = int(digest[:8], 16) % len(ordered)
    return ordered[index]


def _normalize_event_types(
    *,
    events: Optional[Iterable[Dict[str, Any]]] = None,
    event_types: Optional[Iterable[str]] = None,
    valid_event_types: Optional[Sequence[str]] = None,
) -> List[str]:
    normalized = extract_event_types_from_events(
        events or [],
        valid_event_types=valid_event_types,
    )
    if normalized:
        return normalized

    whitelist = {str(item) for item in (valid_event_types or []) if item}
    resolved: List[str] = []
    for event_type in event_types or []:
        current = str(event_type).strip()
        if not current:
            continue
        if whitelist and current not in whitelist:
            continue
        if current not in resolved:
            resolved.append(current)
    return resolved


def build_event_type_response(
    *,
    events: Optional[Iterable[Dict[str, Any]]] = None,
    event_types: Optional[Iterable[str]] = None,
    valid_event_types: Optional[Sequence[str]] = None,
) -> str:
    resolved_types = _normalize_event_types(
        events=events,
        event_types=event_types,
        valid_event_types=valid_event_types,
    )
    payload = [{"event_type": event_type, "arguments": []} for event_type in resolved_types]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_event_type_negative_response(
    *,
    sample_id: str,
    events: Optional[Iterable[Dict[str, Any]]] = None,
    event_types: Optional[Iterable[str]] = None,
    valid_event_types: Optional[Sequence[str]] = None,
) -> str:
    gold_types = _normalize_event_types(
        events=events,
        event_types=event_types,
        valid_event_types=valid_event_types,
    )
    universe = [str(item) for item in (valid_event_types or []) if str(item).strip()]

    if not gold_types:
        if not universe:
            return "[]"
        return build_event_type_response(
            event_types=[_stable_choice(universe, f"{sample_id}:negative-empty")],
        )

    alternatives = [item for item in universe if item not in gold_types]
    negative_types: List[str] = []
    if alternatives:
        for idx, event_type in enumerate(gold_types):
            replacement = _stable_choice(alternatives, f"{sample_id}:{event_type}:{idx}")
            if replacement not in negative_types:
                negative_types.append(replacement)
    else:
        negative_types = gold_types[:-1]

    if negative_types == gold_types:
        negative_types = negative_types[:-1]
    if negative_types == gold_types:
        negative_types = []

    return build_event_type_response(event_types=negative_types)


def _filter_events_by_type(events: Sequence[Dict[str, Any]], event_type: str) -> List[Dict[str, Any]]:
    return [
        dict(event)
        for event in events
        if isinstance(event, dict) and str(event.get("event_type", "")).strip() == event_type
    ]


def _exclude_self_from_fewshot_pool(
    fewshot_example_pool: Optional[Iterable[Dict[str, Any]]],
    *,
    sample_id: str,
) -> List[Dict[str, Any]]:
    if fewshot_example_pool is None:
        return []
    suffix = f":{sample_id}"
    selected: List[Dict[str, Any]] = []
    for example in fewshot_example_pool:
        example_id = str(example.get("id", ""))
        if example_id == sample_id or example_id.endswith(suffix):
            continue
        selected.append(dict(example))
    return selected


def expand_training_samples(
    samples: Sequence[Any],
    *,
    tokenizer: Any,
    schema: Optional[Dict[str, Sequence[str]]] = None,
    stage_mode: str = "single_pass",
    prompt_variant: str = "zeroshot",
    fewshot_num_examples: int = 0,
    fewshot_selection_mode: str = "static",
    fewshot_example_pool: Optional[Iterable[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    resolved_stage_mode = validate_stage_mode(stage_mode)
    normalized_schema = {
        str(event_type): [str(role) for role in list(roles or [])]
        for event_type, roles in (schema or {}).items()
        if event_type
    }
    valid_event_types = list(normalized_schema.keys())
    expanded: List[Dict[str, Any]] = []

    for sample in samples:
        sample_id = str(getattr(sample, "id", ""))
        text = str(getattr(sample, "text", ""))
        events = list(getattr(sample, "events", []) or [])
        event_types = _normalize_event_types(
            events=events,
            event_types=getattr(sample, "event_types", []),
            valid_event_types=valid_event_types,
        )
        sample_pool = _exclude_self_from_fewshot_pool(
            fewshot_example_pool,
            sample_id=sample_id,
        )

        if is_two_stage_mode(resolved_stage_mode):
            stage1_payload = ChinesePromptBuilder.build_event_type_payload(
                text=text,
                schema=normalized_schema,
                tokenizer=tokenizer,
            )
            expanded.append(
                {
                    "sample_id": sample_id,
                    "text": text,
                    "event_types": event_types,
                    "messages": list(stage1_payload.get("messages", [])),
                    "prompt": stage1_payload["formatted_text"],
                    "chosen": build_event_type_response(
                        events=events,
                        event_types=event_types,
                        valid_event_types=valid_event_types,
                    ),
                    "rejected": build_event_type_negative_response(
                        sample_id=sample_id,
                        events=events,
                        event_types=event_types,
                        valid_event_types=valid_event_types,
                    ),
                    "training_stage": "stage1_event_type",
                    "stage_mode": resolved_stage_mode,
                    "fewshot_example_ids": [],
                    "fewshot_selection_mode": "none",
                    "fewshot_count": 0,
                    "lans_eligible": False,
                    "use_precomputed_rejected": True,
                    "stage2_schema_event_types": event_types,
                }
            )
            stage2_targets = (
                [[event_type] for event_type in event_types]
                if resolved_stage_mode == "two_stage_per_type"
                else [event_types]
            )
            for target_event_types in stage2_targets:
                stage2_schema, stage2_schema_event_types = restrict_schema_to_event_types(
                    normalized_schema,
                    target_event_types,
                )
                payload = build_inference_prompt_payload(
                    text=text,
                    tokenizer=tokenizer,
                    prompt_variant=prompt_variant,
                    schema=stage2_schema,
                    num_examples=fewshot_num_examples,
                    fewshot_selection_mode=fewshot_selection_mode,
                    fewshot_example_pool=sample_pool,
                    target_event_types=target_event_types,
                )
                if resolved_stage_mode == "two_stage_per_type":
                    chosen_events = _filter_events_by_type(events, target_event_types[0])
                    chosen_text = json.dumps(chosen_events, ensure_ascii=False, indent=2)
                    record_event_types = list(target_event_types)
                else:
                    chosen_text = str(getattr(sample, "chosen", ""))
                    record_event_types = event_types
                expanded.append(
                    {
                        "sample_id": sample_id,
                        "text": text,
                        "event_types": record_event_types,
                        "messages": list(payload.get("messages", [])),
                        "prompt": payload["formatted_text"],
                        "chosen": chosen_text,
                        "rejected": str(getattr(sample, "rejected", "")),
                        "training_stage": "stage2_extraction",
                        "stage_mode": resolved_stage_mode,
                        "fewshot_example_ids": list(payload.get("fewshot_example_ids", [])),
                        "fewshot_selection_mode": str(payload.get("fewshot_selection_mode", "none")),
                        "fewshot_count": int(payload.get("fewshot_count", 0)),
                        "lans_eligible": True,
                        "use_precomputed_rejected": False,
                        "target_event_types": list(target_event_types),
                        "stage2_schema_event_types": stage2_schema_event_types,
                    }
                )
            continue

        payload = build_inference_prompt_payload(
            text=text,
            tokenizer=tokenizer,
            prompt_variant=prompt_variant,
            schema=normalized_schema,
            num_examples=fewshot_num_examples,
            fewshot_selection_mode=fewshot_selection_mode,
            fewshot_example_pool=sample_pool,
        )
        expanded.append(
            {
                "sample_id": sample_id,
                "text": text,
                "event_types": event_types,
                "messages": list(payload.get("messages", [])),
                "prompt": payload["formatted_text"],
                "chosen": str(getattr(sample, "chosen", "")),
                "rejected": str(getattr(sample, "rejected", "")),
                "training_stage": "single_pass_extraction",
                "stage_mode": resolved_stage_mode,
                "fewshot_example_ids": list(payload.get("fewshot_example_ids", [])),
                "fewshot_selection_mode": str(payload.get("fewshot_selection_mode", "none")),
                "fewshot_count": int(payload.get("fewshot_count", 0)),
                "lans_eligible": True,
                "use_precomputed_rejected": False,
                "stage2_schema_event_types": list(normalized_schema.keys()),
            }
        )

    return expanded


def build_training_cache_metadata(
    *,
    dataset_name: str,
    training_mode: str,
    stage_mode: str,
    prompt_variant: str,
    fewshot_num_examples: int,
    fewshot_selection_mode: str,
    fewshot_pool_split: str,
    research_split_manifest_hash: Optional[str],
    prompt_builder_version: str,
    parser_version: str,
    normalization_version: str,
    model_profile: str,
    max_seq_length: int,
    use_lans: bool,
    taxonomy_hash: Optional[str],
) -> Dict[str, Any]:
    return {
        "cache_version": 1,
        "dataset_name": str(dataset_name),
        "training_mode": str(training_mode),
        "stage_mode": str(stage_mode),
        "prompt_variant": str(prompt_variant),
        "fewshot_num_examples": int(fewshot_num_examples),
        "fewshot_selection_mode": str(fewshot_selection_mode),
        "fewshot_pool_split": str(fewshot_pool_split),
        "research_split_manifest_hash": research_split_manifest_hash,
        "prompt_builder_version": str(prompt_builder_version),
        "parser_version": str(parser_version),
        "normalization_version": str(normalization_version),
        "model_profile": str(model_profile),
        "max_seq_length": int(max_seq_length),
        "use_lans": bool(use_lans),
        "taxonomy_hash": taxonomy_hash,
    }


def training_cache_metadata_matches(expected: Dict[str, Any], observed: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(observed, dict):
        return False
    for key, value in expected.items():
        if observed.get(key) != value:
            return False
    return True
