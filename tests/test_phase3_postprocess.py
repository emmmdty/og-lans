import importlib


def test_postprocess_drops_invalid_event_types_and_rewrites_alias_roles():
    module = importlib.import_module("oglans.utils.json_parser")

    source_text = "2024年1月，阿里巴巴集团中标云平台项目。"
    schema = {"中标": ["中标公司", "中标标的", "中标日期"]}
    alias_map = {"中标": {"中标方": "中标公司"}}
    pred_events = [
        {
            "event_type": "中标",
            "arguments": [
                {"role": "中标方", "argument": "阿里巴巴集团"},
                {"role": "错误角色", "argument": "无效值"},
            ],
        },
        {
            "event_type": "未知事件",
            "arguments": [{"role": "中标公司", "argument": "阿里巴巴集团"}],
        },
    ]

    normalized, diagnostics = module.postprocess_event_list(
        pred_events,
        source_text=source_text,
        schema=schema,
        role_alias_map=alias_map,
    )

    assert normalized == [
        {
            "event_type": "中标",
            "arguments": [{"role": "中标公司", "argument": "阿里巴巴集团"}],
        }
    ]
    assert diagnostics["illegal_event_types_removed"] == 1
    assert diagnostics["illegal_roles_removed"] == 1
    assert diagnostics["alias_rewrites"] == 1
    assert diagnostics["grounding_breakdown"]["exact"] == 1


def test_postprocess_splits_multi_value_argument_only_when_all_parts_are_grounded():
    module = importlib.import_module("oglans.utils.json_parser")

    source_text = "阿里巴巴集团、腾讯中标云平台项目。"
    schema = {"中标": ["中标公司", "中标标的"]}
    pred_events = [
        {
            "event_type": "中标",
            "arguments": [{"role": "中标公司", "argument": "阿里巴巴集团、腾讯"}],
        }
    ]

    normalized, diagnostics = module.postprocess_event_list(
        pred_events,
        source_text=source_text,
        schema=schema,
        role_alias_map={},
    )

    assert normalized[0]["arguments"] == [
        {"role": "中标公司", "argument": "阿里巴巴集团"},
        {"role": "中标公司", "argument": "腾讯"},
    ]
    assert diagnostics["duplicate_splits"] == 1
    assert diagnostics["duplicate_split_unsafe"] == 0
    assert diagnostics["grounding_breakdown"]["exact"] == 2


def test_postprocess_keeps_unsafe_multi_value_argument_and_marks_diagnostic():
    module = importlib.import_module("oglans.utils.json_parser")

    source_text = "阿里巴巴集团、腾讯中标云平台项目。"
    schema = {"中标": ["中标公司", "中标标的"]}
    pred_events = [
        {
            "event_type": "中标",
            "arguments": [{"role": "中标公司", "argument": "阿里巴巴集团、字节跳动"}],
        }
    ]

    normalized, diagnostics = module.postprocess_event_list(
        pred_events,
        source_text=source_text,
        schema=schema,
        role_alias_map={},
    )

    assert normalized[0]["arguments"] == [
        {"role": "中标公司", "argument": "阿里巴巴集团、字节跳动"}
    ]
    assert diagnostics["duplicate_splits"] == 0
    assert diagnostics["duplicate_split_unsafe"] == 1
    assert diagnostics["scv_lite_triggered"] is True
    assert "grounding_failed" in diagnostics["scv_lite_reasons"]


def test_postprocess_retains_ungrounded_argument_but_flags_scv_lite_trigger():
    module = importlib.import_module("oglans.utils.json_parser")

    source_text = "公司公告称完成融资。"
    schema = {"企业融资": ["被投资方", "融资金额"]}
    pred_events = [
        {
            "event_type": "企业融资",
            "arguments": [{"role": "被投资方", "argument": "不存在公司"}],
        }
    ]

    normalized, diagnostics = module.postprocess_event_list(
        pred_events,
        source_text=source_text,
        schema=schema,
        role_alias_map={},
    )

    assert normalized == pred_events
    assert diagnostics["grounded_arguments"] == 0
    assert diagnostics["ungrounded_arguments"] == 1
    assert diagnostics["grounding_breakdown"]["ungrounded"] == 1
    assert diagnostics["scv_lite_triggered"] is True
    assert diagnostics["scv_lite_reasons"] == ["grounding_failed"]
