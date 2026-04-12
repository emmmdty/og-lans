import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


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
build_primary_metric_map = evaluate_module.build_primary_metric_map


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


def test_doc_role_micro_f1_preserves_duplicate_event_records():
    evaluator = AcademicEventEvaluator()
    pred = [
        {
            "event_type": "股东减持",
            "trigger": "减持",
            "arguments": [
                {"role": "减持方", "argument": "张三"},
                {"role": "股票简称", "argument": "远航股份"},
            ],
        }
    ]
    gold = pred + [
        {
            "event_type": "股东减持",
            "trigger": "减持",
            "arguments": [
                {"role": "减持方", "argument": "张三"},
                {"role": "股票简称", "argument": "远航股份"},
            ],
        }
    ]

    evaluator.update(pred, gold)
    report = evaluator.compute_metrics()

    assert report.strict_f1 == pytest.approx(1.0)
    assert report.doc_ee["overall"]["MicroF1"] == pytest.approx(2.0 / 3.0, rel=1e-6)
    assert report.doc_ee["overall"]["TP"] == 2
    assert report.doc_ee["overall"]["FN"] == 2


def test_doc_role_micro_f1_reports_gold_event_multiplicity_breakdown():
    evaluator = AcademicEventEvaluator()
    single_gold = [
        {
            "event_type": "企业融资",
            "arguments": [{"role": "融资金额", "argument": "10亿元"}],
        }
    ]
    multi_gold = [
        {
            "event_type": "企业收购",
            "arguments": [{"role": "收购方", "argument": "甲公司"}],
        },
        {
            "event_type": "企业融资",
            "arguments": [{"role": "融资金额", "argument": "10亿元"}],
        },
    ]
    multi_pred = [
        {
            "event_type": "企业收购",
            "arguments": [{"role": "收购方", "argument": "甲公司"}],
        },
        {
            "event_type": "企业融资",
            "arguments": [{"role": "融资金额", "argument": "20亿元"}],
        },
    ]
    zero_gold_pred = [
        {
            "event_type": "企业融资",
            "arguments": [{"role": "融资金额", "argument": "1亿元"}],
        }
    ]

    evaluator.update(single_gold, single_gold)
    evaluator.update(multi_pred, multi_gold)
    evaluator.update(zero_gold_pred, [])
    report = evaluator.compute_metrics()
    metric_map = build_primary_metric_map(report)
    breakdown = report.doc_ee["gold_event_multiplicity_breakdown"]

    assert breakdown["single_event"]["support_samples"] == 1
    assert breakdown["single_event"]["support_gold_events"] == 1
    assert breakdown["single_event"]["doc_role"]["MicroF1"] == pytest.approx(1.0)
    assert breakdown["multi_event"]["support_samples"] == 1
    assert breakdown["multi_event"]["support_gold_events"] == 2
    assert breakdown["multi_event"]["doc_role"]["MicroF1"] == pytest.approx(0.5)
    assert breakdown["zero_gold"]["support_samples"] == 1
    assert metric_map["single_event_doc_role_micro_f1"] == pytest.approx(1.0)
    assert metric_map["multi_event_doc_role_micro_f1"] == pytest.approx(0.5)


def test_text_proxy_metrics_and_grounding_coverage_are_reported():
    evaluator = AcademicEventEvaluator()
    pred = [
        {
            "event_type": "企业收购",
            "trigger": "收购",
            "arguments": [
                {"role": "收购方", "argument": "沙特阿美"},
                {"role": "被收购方", "argument": "SABIC"},
            ],
        }
    ]

    evaluator.update_with_extended_metrics(
        pred_events=pred,
        gold_events=pred,
        source_text="沙特阿美正在收购SABIC。",
        full_response="",
        parse_success=True,
        parse_diagnostics={"success": True, "repair_steps": []},
        valid_event_types={"企业收购"},
        valid_roles_by_event={"企业收购": {"收购方", "被收购方"}},
    )
    report = evaluator.compute_metrics()

    assert report.ee_text_proxy["trigger_text_cls"]["f1"] == pytest.approx(1.0)
    assert report.ee_text_proxy["argument_attached_text_cls"]["f1"] == pytest.approx(1.0)
    assert report.ee_text_proxy["grounding_coverage"] == pytest.approx(1.0)


def test_primary_metric_map_exposes_legacy_dueefin_track():
    evaluator = AcademicEventEvaluator()
    pred = [
        {
            "event_type": "企业融资",
            "arguments": [{"role": "融资金额", "argument": "10亿元"}],
        }
    ]
    gold = pred

    evaluator.update(pred, gold)
    report = evaluator.compute_metrics()
    metric_map = build_primary_metric_map(report)

    assert metric_map["legacy_dueefin_overall_precision"] == pytest.approx(1.0)
    assert metric_map["legacy_dueefin_overall_recall"] == pytest.approx(1.0)
    assert metric_map["legacy_dueefin_overall_f1"] == pytest.approx(1.0)
    assert metric_map["legacy_dueefin_overall_f1"] == pytest.approx(metric_map["strict_f1"])
