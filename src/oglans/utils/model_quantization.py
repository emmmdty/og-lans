"""
Helpers for model quantization/runtime behavior detection.
"""

from __future__ import annotations

from typing import Any


def is_quantized_model(model: Any) -> bool:
    """
    判断模型是否为 bitsandbytes 量化模型（4bit/8bit）。
    """
    if model is None:
        return False

    for attr in ("is_loaded_in_4bit", "is_loaded_in_8bit", "is_quantized"):
        try:
            if bool(getattr(model, attr, False)):
                return True
        except Exception:
            pass

    quantization_config = getattr(model, "quantization_config", None)
    if quantization_config is not None:
        try:
            if bool(getattr(quantization_config, "load_in_4bit", False)):
                return True
            if bool(getattr(quantization_config, "load_in_8bit", False)):
                return True
            if hasattr(quantization_config, "bnb_4bit_quant_type"):
                return True
        except Exception:
            pass

    return False
