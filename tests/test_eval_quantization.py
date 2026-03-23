import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "oglans" / "utils" / "model_quantization.py"
spec = importlib.util.spec_from_file_location("model_quant_module", str(MODULE_PATH))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load module from {MODULE_PATH}")
quant_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quant_module)


class _DummyQ4:
    is_loaded_in_4bit = True


class _DummyQCfg:
    class QCfg:
        load_in_8bit = True

    quantization_config = QCfg()


class _DummyFP:
    pass


def test_is_quantized_model_4bit_flag():
    assert quant_module.is_quantized_model(_DummyQ4()) is True


def test_is_quantized_model_quantization_config():
    assert quant_module.is_quantized_model(_DummyQCfg()) is True


def test_is_quantized_model_false_for_plain_model():
    assert quant_module.is_quantized_model(_DummyFP()) is False
