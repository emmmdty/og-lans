#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OG-LANS Configuration Manager

Centralized configuration management for the OG-LANS training framework.
Implements a thread-safe singleton pattern for global configuration access.

Features:
    - YAML configuration file loading
    - CLI parameter override support (e.g., --training.max_steps 500)
    - Nested key path resolution (dot notation)
    - Type-safe value parsing

Configuration Structure:
    project:
        output_dir, logging_dir, seed, etc.
    model:
        base_model, max_seq_length, load_in_4bit, etc.
    training:
        learning_rate, num_train_epochs, batch_size, etc.
    algorithms:
        lans: LANS scheduler parameters
        ds_cns: Ontology graph sampler settings
        scv: Semantic consistency verification settings

Example:
    >>> from oglans.config import ConfigManager
    >>> config = ConfigManager.load_config("configs/config.yaml", ["--training.max_steps", "100"])
    >>> print(config['training']['max_steps'])
    100

Authors:
    OG-LANS Research Team
"""

import copy
import os
from pathlib import Path
import yaml
import logging
from typing import Dict, Any, Optional

from oglans.utils.experiment_contract import normalize_postprocess_profile
from oglans.utils.training_protocol import resolve_training_resume_settings

logger = logging.getLogger("OGLANS")


SEMANTIC_REQUIRED_PATHS = (
    "model.profile",
    "comparison.prompt_builder_version",
    "comparison.parser_version",
    "comparison.normalization_version",
    "evaluation.mode",
)

class ConfigManager:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def load_config(
        cls,
        config_path: str = "configs/config.yaml",
        overrides: list = None,
        *,
        validate_semantic: bool = True,
    ) -> Dict[str, Any]:
        """
        加载配置文件并应用覆盖
        """
        config = cls._load_config_file(config_path)

        if overrides:
            cls._apply_cli_overrides(config, overrides)

        cls._apply_runtime_defaults(config)
        if validate_semantic:
            cls._validate_semantic_contract(config)

        cls._config = config
        return config

    @classmethod
    def _load_config_file(cls, config_path: str, visited: Optional[set[str]] = None) -> Dict[str, Any]:
        path_obj = Path(config_path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        resolved = str(path_obj.resolve())
        visited = visited or set()
        if resolved in visited:
            raise ValueError(f"Detected recursive config extends: {resolved}")
        visited.add(resolved)

        with path_obj.open('r', encoding='utf-8') as f:
            payload = yaml.safe_load(f) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Configuration root must be a dict: {config_path}")

        parent_ref = payload.pop("extends", None)
        if not parent_ref:
            return payload

        parent_path = Path(parent_ref)
        if not parent_path.is_absolute():
            parent_path = path_obj.parent / parent_path
        parent_config = cls._load_config_file(str(parent_path), visited=visited)
        return cls._deep_merge_dict(parent_config, payload)

    @staticmethod
    def _deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = copy.deepcopy(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = ConfigManager._deep_merge_dict(merged[key], value)
            else:
                merged[key] = copy.deepcopy(value)
        return merged

    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        if cls._config is None:
            raise RuntimeError("Config not loaded. Call load_config() first.")
        return cls._config

    @staticmethod
    def _apply_cli_overrides(config: dict, overrides: list) -> None:
        """
        处理命令行覆盖
        """
        if not overrides:
            return

        if len(overrides) % 2 != 0:
            raise ValueError("Command line overrides must be in pairs: --key value")

        for i in range(0, len(overrides), 2):
            key = overrides[i]
            value = overrides[i + 1]
            if not key.startswith("--"):
                continue

            path = key[2:].split(".")
            # Parse value
            try:
                parsed_value = yaml.safe_load(value)
            except Exception:
                parsed_value = value

            cur = config
            for p in path[:-1]:
                if p not in cur or not isinstance(cur[p], dict):
                    cur[p] = {}
                cur = cur[p]
            cur[path[-1]] = parsed_value

    @staticmethod
    def _apply_runtime_defaults(config: dict) -> None:
        """
        为运行时关键字段补默认值，兼容历史配置文件。
        """
        algorithms = config.setdefault("algorithms", {})
        lans_cfg = algorithms.setdefault("lans", {})
        scv_cfg = algorithms.setdefault("scv", {})

        lans_cfg.setdefault("refresh_start_epoch", 1)
        lans_cfg.setdefault("refresh_log_interval", 200)
        lans_cfg.setdefault("refresh_log_seconds", 30)
        lans_cfg.setdefault("loss_baseline", 0.5)
        lans_cfg.setdefault("runtime_mode", "online_iterable")
        lans_cfg.setdefault("signal_center", lans_cfg.get("loss_baseline", 0.5))
        lans_cfg.setdefault("signal_temperature", 0.25)
        lans_strategies = lans_cfg.setdefault("strategies", {})
        lans_strategies.setdefault("easy_ratio", 0.7)
        lans_strategies.setdefault("hard_ratio", 0.4)
        lans_strategies.setdefault("hard_floor_prob", 0.0)
        lans_strategies.setdefault("hard_floor_after_warmup", None)
        lans_strategies.setdefault("medium_floor_prob", 0.0)

        scv_cfg.setdefault("progress_log_interval", 200)
        scv_cfg.setdefault("progress_log_seconds", 30)
        scv_cfg.setdefault("cache_enabled", True)
        scv_cfg.setdefault("cache_max_entries", 50000)
        scv_cfg.setdefault("max_retries", 1)
        scv_cfg.setdefault("source", "modelscope")
        scv_cfg.setdefault("entailment_idx", None)

        training_cfg = config.setdefault("training", {})
        training_cfg.setdefault("mode", "preference")
        training_cfg.setdefault("fast_io", False)
        training_cfg.setdefault("dataloader_num_workers", 0)
        training_cfg.setdefault("dataloader_pin_memory", False)
        training_cfg.setdefault("aux_log_interval", 50)
        training_cfg.setdefault("resume_training_from", None)
        training_cfg.setdefault("warm_start_from_checkpoint", None)
        teacher_silver_cfg = training_cfg.setdefault("teacher_silver", {})
        teacher_silver_cfg.setdefault("enabled", False)
        teacher_silver_cfg.setdefault("path", None)
        teacher_silver_cfg.setdefault("max_samples", None)
        teacher_silver_cfg.setdefault("id_prefix", "teacher")
        rpo_cfg = training_cfg.setdefault("rpo", {})
        rpo_cfg.setdefault("alpha", 0.0)
        rpo_cfg.setdefault("warmup_steps", 0)
        rpo_cfg.setdefault("require_valid_labels", True)
        rpo_cfg.setdefault("log_interval", training_cfg.get("aux_log_interval", 50))
        preference_cfg = training_cfg.setdefault("preference", {})
        preference_cfg.setdefault("mode", "ipo")
        preference_cfg.setdefault("offset_source", "margin_bucket")
        preference_cfg.setdefault("offset_static", 0.15)
        preference_cfg.setdefault("offset_clip_min", 0.0)
        preference_cfg.setdefault("offset_clip_max", 1.0)

        inference_cfg = config.setdefault("inference", {})
        inference_cfg.setdefault("pipeline_mode", "e2e")
        postprocess_cfg = inference_cfg.setdefault("postprocess", {})
        postprocess_cfg.setdefault("enabled", False)
        postprocess_cfg.setdefault("role_whitelist", True)
        postprocess_cfg.setdefault("alias_map", True)
        postprocess_cfg.setdefault("duplicate_role_split", True)
        postprocess_cfg.setdefault("normalization_mode", "diagnostics_only")
        postprocess_cfg.setdefault("grounding_mode", "exact+fuzzy+code_local")
        postprocess_cfg.setdefault("sidecar_diagnostics", True)
        postprocess_cfg.setdefault("preserve_ungrounded_arguments", True)
        scv_lite_cfg = inference_cfg.setdefault("scv_lite", {})
        scv_lite_cfg.setdefault("mode", "off")
        scv_lite_cfg.setdefault("trigger_on_grounding_failure", True)
        scv_lite_cfg.setdefault("trigger_on_mutual_exclusion", False)
        scv_lite_cfg.setdefault("trigger_on_shared_role_conflict", False)

        comparison_cfg = config.setdefault("comparison", {})
        comparison_cfg.setdefault("eval_protocol_path", "./configs/eval_protocol.yaml")
        comparison_cfg.setdefault("role_alias_map_path", "./configs/role_aliases_duee_fin.yaml")
        comparison_cfg.setdefault("prompt_variant", "zeroshot")
        comparison_cfg.setdefault("fewshot_num_examples", 3)
        comparison_cfg.setdefault("stage_mode", "single_pass")
        comparison_cfg.setdefault("fewshot_selection_mode", "dynamic")
        comparison_cfg.setdefault("fewshot_pool_split", "train_fit")
        comparison_cfg.setdefault("train_tune_ratio", 0.1)
        comparison_cfg.setdefault("research_split_manifest_path", None)
        comparison_cfg.setdefault("postprocess_profile", "none")

    @staticmethod
    def _get_nested(config: Dict[str, Any], path: str) -> Any:
        cur: Any = config
        for key in path.split("."):
            if not isinstance(cur, dict) or key not in cur:
                return None
            cur = cur[key]
        return cur

    @classmethod
    def _validate_semantic_contract(cls, config: Dict[str, Any]) -> None:
        missing = [path for path in SEMANTIC_REQUIRED_PATHS if cls._get_nested(config, path) is None]
        if missing:
            raise ValueError(
                "Missing required semantic config fields: " + ", ".join(missing)
            )
        model_source = str(cls._get_nested(config, "model.source") or "").strip().lower()
        if model_source not in {"local", "modelscope"}:
            raise ValueError(
                f"Unsupported model.source: {model_source}. "
                "Expected one of local, modelscope."
            )
        scv_source = str(cls._get_nested(config, "algorithms.scv.source") or "").strip().lower()
        if scv_source and scv_source not in {"local", "modelscope"}:
            raise ValueError(
                f"Unsupported algorithms.scv.source: {scv_source}. "
                "Expected one of local, modelscope."
            )
        evaluation_mode = str(cls._get_nested(config, "evaluation.mode") or "").strip().lower()
        if evaluation_mode not in {"scored", "prediction_only"}:
            raise ValueError(
                f"Unsupported evaluation.mode: {evaluation_mode}. "
                "Expected one of scored, prediction_only."
            )
        resolve_training_resume_settings(config.get("training", {}) or {})
        if cls._get_nested(config, "api_evaluation.system_prompt_style") is not None:
            raise ValueError(
                "api_evaluation.system_prompt_style is no longer supported. "
                "Prompt style is determined by model.profile."
            )
        if cls._get_nested(config, "comparison.enable_thinking") is not None:
            raise ValueError(
                "comparison.enable_thinking is no longer supported. "
                "The current prompt contract requires plain JSON output without hidden reasoning toggles."
            )
        if cls._get_nested(config, "comparison.thinking_budget") is not None:
            raise ValueError(
                "comparison.thinking_budget is no longer supported. "
                "The current prompt contract requires plain JSON output without hidden reasoning budgets."
            )
        normalize_postprocess_profile(cls._get_nested(config, "comparison.postprocess_profile"))
