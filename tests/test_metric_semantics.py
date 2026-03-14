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


def test_parse_diagnostics_split_raw_vs_repair_vs_extraction_fail():
    evaluator = AcademicEventEvaluator()
    pred = [{"event_type": "质押", "arguments": [{"role": "质押方", "argument": "张三"}]}]
    gold = pred

    evaluator.update_with_extended_metrics(
        pred_events=pred,
        gold_events=gold,
        source_text="张三办理质押。",
        full_response="",
        parse_success=True,
        parse_diagnostics={"success": True, "repair_steps": []},
        valid_event_types={"质押"},
        valid_roles_by_event={"质押": {"质押方"}},
    )
    evaluator.update_with_extended_metrics(
        pred_events=pred,
        gold_events=gold,
        source_text="张三办理质押。",
        full_response="",
        parse_success=True,
        parse_diagnostics={"success": True, "repair_steps": ["fix_trailing_comma"]},
        valid_event_types={"质押"},
        valid_roles_by_event={"质押": {"质押方"}},
    )
    evaluator.update_with_extended_metrics(
        pred_events=[],
        gold_events=gold,
        source_text="张三办理质押。",
        full_response="",
        parse_success=False,
        parse_diagnostics={"success": False, "extraction_method": "no_json_found"},
        valid_event_types={"质押"},
        valid_roles_by_event={"质押": {"质押方"}},
    )
    report = evaluator.compute_metrics()
    assert report.total_samples == 3
    assert report.parse_raw_success == 1
    assert report.parse_repair_success == 1
    assert report.parse_errors == 1
    assert report.parse_extraction_failures == 1


def test_schema_and_hallucination_breakdown_populated():
    evaluator = AcademicEventEvaluator()
    pred = [{"event_type": "质押", "arguments": [{"role": "非法角色", "argument": "不存在实体"}]}]
    gold = []
    evaluator.update_with_extended_metrics(
        pred_events=pred,
        gold_events=gold,
        source_text="这是一条没有相关实体的文本。",
        full_response="<thought>事件类型：质押\n非法角色 = 不存在实体</thought>",
        parse_success=True,
        parse_diagnostics={"success": True, "repair_steps": []},
        valid_event_types={"质押"},
        valid_roles_by_event={"质押": {"质押方"}},
    )
    report = evaluator.compute_metrics()
    assert report.schema_compliance_rate == 0.0
    assert any(k.startswith("invalid_role:") for k in report.schema_violation_breakdown)
    assert len(report.hallucination_breakdown) >= 1
