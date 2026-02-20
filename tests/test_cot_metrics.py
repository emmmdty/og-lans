import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if "dirtyjson" not in sys.modules:
    sys.modules["dirtyjson"] = SimpleNamespace(loads=json.loads)

EVAL_PATH = ROOT / "evaluate.py"
spec = importlib.util.spec_from_file_location("evaluate_module", str(EVAL_PATH))
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load evaluate module from {EVAL_PATH}")
evaluate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_module)

AcademicEventEvaluator = evaluate_module.AcademicEventEvaluator


def test_cot_missing_thought_is_skipped_not_true():
    evaluator = AcademicEventEvaluator()
    pred = [{"event_type": "质押", "arguments": [{"role": "质押方", "argument": "张三"}]}]
    gold = pred
    response = "```json\n[{\"event_type\":\"质押\",\"arguments\":[{\"role\":\"质押方\",\"argument\":\"张三\"}]}]\n```"
    evaluator.update_with_extended_metrics(
        pred_events=pred,
        gold_events=gold,
        source_text="张三办理质押。",
        full_response=response,
        parse_success=True,
        parse_diagnostics={"success": True, "repair_steps": []},
        valid_event_types={"质押"},
        valid_roles_by_event={"质押": {"质押方"}},
    )
    report = evaluator.compute_metrics()
    assert report.cot_checked == 0
    assert report.cot_skipped == 1
    assert report.cot_faithfulness == 0.0


def test_cot_type_inconsistency_is_detected():
    evaluator = AcademicEventEvaluator()
    pred = [{"event_type": "质押", "arguments": [{"role": "质押方", "argument": "张三"}]}]
    gold = pred
    response = (
        "<thought>\n"
        "第一步：事件类型为股份回购\n"
        "第二步：回购方 = \"张三\"\n"
        "</thought>\n"
        "```json\n[{\"event_type\":\"质押\",\"arguments\":[{\"role\":\"质押方\",\"argument\":\"张三\"}]}]\n```"
    )
    evaluator.update_with_extended_metrics(
        pred_events=pred,
        gold_events=gold,
        source_text="张三办理质押。",
        full_response=response,
        parse_success=True,
        parse_diagnostics={"success": True, "repair_steps": []},
        valid_event_types={"质押"},
        valid_roles_by_event={"质押": {"质押方"}},
    )
    report = evaluator.compute_metrics()
    assert report.cot_checked == 1
    assert report.cot_type_consistency == 0.0
    assert report.cot_faithfulness == 0.0


def test_cot_argument_inconsistency_is_detected():
    evaluator = AcademicEventEvaluator()
    pred = [{"event_type": "质押", "arguments": [{"role": "质押方", "argument": "张三"}]}]
    gold = pred
    response = (
        "<thought>\n"
        "第一步：事件类型为质押\n"
        "第二步：质押方 = \"李四\"\n"
        "</thought>\n"
        "```json\n[{\"event_type\":\"质押\",\"arguments\":[{\"role\":\"质押方\",\"argument\":\"张三\"}]}]\n```"
    )
    evaluator.update_with_extended_metrics(
        pred_events=pred,
        gold_events=gold,
        source_text="张三办理质押。",
        full_response=response,
        parse_success=True,
        parse_diagnostics={"success": True, "repair_steps": []},
        valid_event_types={"质押"},
        valid_roles_by_event={"质押": {"质押方"}},
    )
    report = evaluator.compute_metrics()
    assert report.cot_checked == 1
    assert report.cot_type_consistency == 1.0
    assert report.cot_argument_consistency == 0.0
    assert report.cot_faithfulness == 0.0
