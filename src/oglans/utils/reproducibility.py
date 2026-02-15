#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OG-LANS Reproducibility Utilities

Ensures experimental reproducibility across different runs and hardware.

This module provides comprehensive seed management for:
    - Python's random module
    - NumPy random number generators
    - PyTorch CPU and CUDA operations
    - Hugging Face Transformers

Academic Reproducibility:
    Following best practices from "Reproducibility in Machine Learning" literature,
    this module sets PYTHONHASHSEED and configures cuBLAS workspace for deterministic
    CUDA operations when `deterministic=True`.

Warning:
    Enabling `deterministic=True` may reduce training performance by 10-20%
    due to disabled cuDNN auto-tuning. For final experiments, always enable
    determinism; for development, you may disable it for faster iteration.

Usage:
    >>> from oglans.utils.reproducibility import set_global_seed
    >>> set_global_seed(3407, deterministic=True)

References:
    [1] PyTorch Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
    [2] Bouthillier et al. "Unreproducible Research is Reproducible" ICML 2019

Authors:
    OG-LANS Research Team
"""

import random
import numpy as np
import torch
import os
import logging

logger = logging.getLogger("OGLANS")

def set_global_seed(seed: int = 3407, deterministic: bool = True):
    """
    设置全局随机种子以确保实验可复现性

    Args:
        seed: 随机种子
        deterministic: 是否启用 cudnn 确定性模式 (会降低性能但保证完全一致)
    """
    logger.info(f"🔒 Setting global seed to {seed} (deterministic={deterministic})")

    # Python stdlib
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

    # Transformer set_seed (redundant but safe)
    from transformers import set_seed
    set_seed(seed)

    # Deterministic operations
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 某些 PyTorch 操作可能仍不确定，但在 DPO 场景下这通常足够
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # 某些操作需要此配置
