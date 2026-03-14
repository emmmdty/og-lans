from oglans.inference.cat_lite import apply_cat_lite_pipeline, perturb_text_for_counterfactual


def test_cat_lite_filters_invalid_events_and_arguments():
    source_text = "2024年1月，阿里巴巴集团宣布回购100亿元股份。"
    pred_events = [
        {
            "event_type": "股份回购",
            "arguments": [
                {"role": "回购方", "argument": "阿里巴巴集团"},
                {"role": "交易金额", "argument": "100亿元"},
                {"role": "错误角色", "argument": "无效"},
            ],
        },
        {
            "event_type": "未知类型",
            "arguments": [{"role": "回购方", "argument": "阿里巴巴集团"}],
        },
    ]
    schema = {
        "股份回购": ["回购方", "交易金额"],
    }

    result = apply_cat_lite_pipeline(
        pred_events=pred_events,
        source_text=source_text,
        schema=schema,
        require_argument_in_text=True,
    )

    assert len(result.events) == 1
    assert result.kept_events == 1
    assert result.dropped_events == 1
    assert result.kept_arguments == 2
    assert result.dropped_arguments == 1


def test_perturb_text_for_counterfactual_changes_number_when_present():
    text = "公司宣布回购100亿元股份。"
    perturbed, meta = perturb_text_for_counterfactual(text, target_types=("number",))

    assert meta.get("changed") is True
    assert meta.get("type") == "number"
    assert "100亿元" not in perturbed
