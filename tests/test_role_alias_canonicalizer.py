import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "oglans" / "utils" / "eval_protocol.py"
spec = importlib.util.spec_from_file_location("eval_protocol_module", str(MODULE_PATH))
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load module from {MODULE_PATH}")
eval_protocol_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_protocol_module)


def test_canonicalize_pred_roles_rewrites_expected_roles():
    pred_events = [
        {
            "event_type": "中标",
            "arguments": [
                {"role": "中标方", "argument": "甲公司"},
                {"role": "中标项目", "argument": "某项目"},
            ],
        }
    ]
    alias_map = {"中标": {"中标方": "中标公司", "中标项目": "中标标的"}}

    normalized, rewritten = eval_protocol_module.canonicalize_pred_roles(pred_events, alias_map)

    assert rewritten == 2
    assert normalized[0]["arguments"][0]["role"] == "中标公司"
    assert normalized[0]["arguments"][1]["role"] == "中标标的"
    # 原输入不应被原地修改
    assert pred_events[0]["arguments"][0]["role"] == "中标方"
