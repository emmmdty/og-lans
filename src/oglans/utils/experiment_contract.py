"""
Shared experiment-contract helpers for training and evaluation entrypoints.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping


LEGACY_STAGE_MODE_ALIASES = {
    "two_stage_typed": "two_stage_per_type",
}
SUPPORTED_STAGE_MODES = ("single_pass", "two_stage", "two_stage_per_type")
STAGE_MODE_CHOICES = SUPPORTED_STAGE_MODES + tuple(LEGACY_STAGE_MODE_ALIASES.keys())
SUPPORTED_POSTPROCESS_PROFILES = ("none", "event_probe_v2")

EXPERIMENT_CONTRACT_FIELDS = (
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
    "postprocess_profile",
    "canonical_metric_mode",
    "prompt_builder_version",
    "configured_prompt_builder_version",
    "parser_version",
    "configured_parser_version",
    "normalization_version",
    "configured_normalization_version",
    "protocol_hash",
    "role_alias_hash",
    "seed",
    "seed_effective",
    "token_usage_kind",
    "experiment_contract_hash",
)

EXPERIMENT_HASH_FIELDS = (
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
    "postprocess_profile",
    "canonical_metric_mode",
    "prompt_builder_version",
    "parser_version",
    "normalization_version",
    "protocol_hash",
    "role_alias_hash",
)

COMPARE_DERIVED_FIELDS = (
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
    "postprocess_profile",
    "canonical_metric_mode",
    "prompt_builder_version",
    "parser_version",
    "normalization_version",
    "protocol_hash",
    "role_alias_hash",
    "seed",
    "seed_effective",
    "token_usage_kind",
)


def _stable_hash(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def normalize_stage_mode(stage_mode: Any) -> str:
    normalized = str(stage_mode or "single_pass").strip().lower()
    normalized = LEGACY_STAGE_MODE_ALIASES.get(normalized, normalized)
    if normalized not in SUPPORTED_STAGE_MODES:
        raise ValueError(
            f"Unsupported stage_mode: {normalized}. "
            f"Expected one of {', '.join(STAGE_MODE_CHOICES)}."
        )
    return normalized


def is_two_stage_mode(stage_mode: Any) -> bool:
    return normalize_stage_mode(stage_mode) in {"two_stage", "two_stage_per_type"}


def normalize_postprocess_profile(profile: Any) -> str:
    normalized = str(profile or "none").strip().lower()
    if normalized not in SUPPORTED_POSTPROCESS_PROFILES:
        raise ValueError(
            f"Unsupported postprocess_profile: {normalized}. "
            f"Expected one of {', '.join(SUPPORTED_POSTPROCESS_PROFILES)}."
        )
    return normalized


def build_experiment_contract(payload: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = {
        key: payload.get(key)
        for key in EXPERIMENT_CONTRACT_FIELDS
        if key != "experiment_contract_hash"
    }
    normalized["stage_mode"] = normalize_stage_mode(normalized.get("stage_mode"))
    normalized["postprocess_profile"] = normalize_postprocess_profile(normalized.get("postprocess_profile"))
    normalized["configured_prompt_builder_version"] = str(
        normalized.get("configured_prompt_builder_version")
        or normalized.get("prompt_builder_version")
        or ""
    )
    normalized["configured_parser_version"] = str(
        normalized.get("configured_parser_version")
        or normalized.get("parser_version")
        or ""
    )
    normalized["configured_normalization_version"] = str(
        normalized.get("configured_normalization_version")
        or normalized.get("normalization_version")
        or ""
    )
    missing = [
        key
        for key, value in normalized.items()
        if value is None
    ]
    if missing:
        raise ValueError(f"Missing experiment-contract fields: {', '.join(sorted(missing))}")
    normalized["experiment_contract_hash"] = _stable_hash(
        {key: normalized[key] for key in EXPERIMENT_HASH_FIELDS}
    )
    return normalized


def build_compare_contract_payload(experiment_contract: Mapping[str, Any]) -> Dict[str, Any]:
    contract = build_experiment_contract(experiment_contract)
    return {key: contract[key] for key in COMPARE_DERIVED_FIELDS}


def extract_experiment_contract(payload: Mapping[str, Any]) -> Dict[str, Any]:
    contract = payload.get("experiment_contract")
    if isinstance(contract, Mapping):
        return build_experiment_contract(contract)
    compare = payload.get("compare")
    if isinstance(compare, Mapping):
        contract_payload = dict(compare)
        meta = payload.get("meta")
        if isinstance(meta, Mapping):
            for key in (
                "postprocess_profile",
                "prompt_builder_version",
                "configured_prompt_builder_version",
                "parser_version",
                "configured_parser_version",
                "normalization_version",
                "configured_normalization_version",
            ):
                if contract_payload.get(key) is None and meta.get(key) is not None:
                    contract_payload[key] = meta.get(key)
        return build_experiment_contract(contract_payload)
    raise ValueError("Missing experiment_contract block")
