#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lazy trainer exports so CPU-only environments can import ``oglans.trainer``
without immediately importing Unsloth/GPU-bound modules.
"""

from __future__ import annotations

from typing import Any

__all__ = ["UnslothDPOTrainerWrapper", "UnslothSFTTrainerWrapper"]

_trainer_import_error: Exception | None = None


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module 'oglans.trainer' has no attribute '{name}'")

    try:
        from .unsloth_trainer import UnslothDPOTrainerWrapper, UnslothSFTTrainerWrapper
    except Exception as exc:  # pragma: no cover - exercised in CPU-only environments
        global _trainer_import_error
        _trainer_import_error = exc
        raise ImportError(
            "oglans.trainer requires the Unsloth training stack and a supported accelerator. "
            "Import the specific wrapper only in a GPU-ready training environment."
        ) from exc

    exported = {
        "UnslothDPOTrainerWrapper": UnslothDPOTrainerWrapper,
        "UnslothSFTTrainerWrapper": UnslothSFTTrainerWrapper,
    }
    globals().update(exported)
    return exported[name]


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
