#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OG-LANS: Ontology-Graph Loss-Aware Adaptive Negative Sampling

A curriculum learning framework for improving Direct Preference Optimization (DPO)
in structured Event Extraction tasks.

Core Components:
    - DuEEFinAdapter: Data adapter for DuEE-Fin financial event extraction dataset
    - UnslothDPOTrainerWrapper: Custom DPO trainer with LANS and CGA integration
    - setup_logger: Unified logging configuration

Submodules:
    - oglans.data: Data loading and prompt building
    - oglans.trainer: Training logic with CGADPOTrainer and LANSCallback
    - oglans.utils: DS-CNS sampler, SCV verifier, and utilities

Example:
    >>> from oglans import DuEEFinAdapter, UnslothDPOTrainerWrapper
    >>> adapter = DuEEFinAdapter("./data/raw/DuEE-Fin", "./schema.json")
    >>> samples = adapter.load_data("train")

Version: 1.0.0
License: MIT
"""

from .data.adapter import DuEEFinAdapter

__version__ = "1.0.0"
__author__ = "OG-LANS Research Team"
__all__ = ["DuEEFinAdapter", "UnslothDPOTrainerWrapper", "setup_logger"]


def __getattr__(name: str):
    if name == "UnslothDPOTrainerWrapper":
        try:
            from .trainer.unsloth_trainer import UnslothDPOTrainerWrapper
            globals()["UnslothDPOTrainerWrapper"] = UnslothDPOTrainerWrapper
            return UnslothDPOTrainerWrapper
        except Exception as exc:  # pragma: no cover - 仅在缺少训练依赖时触发
            raise ImportError(
                "UnslothDPOTrainerWrapper 依赖未安装。请先安装 unsloth/trl 等训练依赖。"
            ) from exc
    if name == "setup_logger":
        try:
            from .utils.logger import setup_logger
            globals()["setup_logger"] = setup_logger
            return setup_logger
        except Exception as exc:  # pragma: no cover - 仅在缺少可选依赖时触发
            raise ImportError(
                "setup_logger 依赖未安装。请补齐 utils 相关依赖（如 networkx 等）。"
            ) from exc
    raise AttributeError(f"module 'oglans' has no attribute '{name}'")
