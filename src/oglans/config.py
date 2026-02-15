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

import os
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
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        if overrides:
            cls._apply_cli_overrides(config, overrides)

        cls._apply_runtime_defaults(config)

        cls._config = config
        return config

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
        scv_cfg.setdefault("progress_log_interval", 200)
        scv_cfg.setdefault("progress_log_seconds", 30)

        training_cfg = config.setdefault("training", {})
        training_cfg.setdefault("fast_io", False)
        training_cfg.setdefault("dataloader_num_workers", 0)
        training_cfg.setdefault("dataloader_pin_memory", False)
