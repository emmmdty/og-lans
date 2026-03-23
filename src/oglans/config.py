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

logger = logging.getLogger("OGLANS")

class ConfigManager:
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    @classmethod
    def load_config(cls, config_path: str = "configs/config.yaml", overrides: list = None) -> Dict[str, Any]:
        """
        加载配置文件并应用覆盖
        """
        config = cls._load_config_file(config_path)

        if overrides:
            cls._apply_cli_overrides(config, overrides)

        cls._apply_runtime_defaults(config)

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

        training_cfg = config.setdefault("training", {})
        training_cfg.setdefault("mode", "preference")
        training_cfg.setdefault("fast_io", False)
        training_cfg.setdefault("dataloader_num_workers", 0)
        training_cfg.setdefault("dataloader_pin_memory", False)
        training_cfg.setdefault("aux_log_interval", 50)
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

        comparison_cfg = config.setdefault("comparison", {})
        comparison_cfg.setdefault("eval_protocol_path", "./configs/eval_protocol.yaml")
        comparison_cfg.setdefault("role_alias_map_path", "./configs/role_aliases_duee_fin.yaml")
        comparison_cfg.setdefault("prompt_variant", "zeroshot")
        comparison_cfg.setdefault("fewshot_num_examples", 3)
        comparison_cfg.setdefault("prompt_builder_version", "route_a_compare_v1")
        comparison_cfg.setdefault("parser_version", "route_a_compare_v1")
        comparison_cfg.setdefault("normalization_version", "route_a_compare_v1")
