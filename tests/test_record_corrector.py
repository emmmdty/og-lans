from oglans.inference.record_corrector import (
    apply_record_corrector,
    apply_structured_event_pipeline,
    validate_pipeline_mode,
)


def test_record_corrector_splits_multi_company_interview_record():
    source_text = "2024年6月19日，监管部门约谈甲公司、乙公司，要求整改。"
    schema = {"被约谈": ["公司名称", "披露时间", "被约谈时间", "约谈机构"]}
    pred_events = [
        {
            "event_type": "被约谈",
            "trigger": "约谈",
            "arguments": [
                {"role": "公司名称", "argument": "甲公司、乙公司"},
                {"role": "约谈机构", "argument": "监管部门"},
                {"role": "披露时间", "argument": "2024年6月19日"},
            ],
        }
    ]

    result = apply_record_corrector(
        pred_events,
        source_text=source_text,
        schema=schema,
        role_alias_map={},
    )

    assert len(result.events) == 2
    assert result.records_split_count == 1
    assert [event["arguments"][0]["argument"] for event in result.events] == ["甲公司", "乙公司"]


def test_record_corrector_adds_buyback_quantity_and_price_when_grounded():
    source_text = (
        "公司公告称拟回购股份，回购价格不超过150元/股。"
        "本次累计回购股份数量500万股，交易金额不低于5亿元。"
    )
    schema = {
        "股份回购": [
            "回购方",
            "披露时间",
            "回购股份数量",
            "每股交易价格",
            "占公司总股本比例",
            "交易金额",
            "回购完成时间",
        ]
    }
    pred_events = [
        {
            "event_type": "股份回购",
            "arguments": [
                {"role": "交易金额", "argument": "不低于5亿元"},
            ],
        }
    ]

    result = apply_record_corrector(
        pred_events,
        source_text=source_text,
        schema=schema,
        role_alias_map={},
    )

    args = result.events[0]["arguments"]
    assert {"role": "回购股份数量", "argument": "500万股"} in args
    assert {"role": "每股交易价格", "argument": "150元/股"} in args
    assert result.roles_added_count >= 2


def test_record_corrector_does_not_add_ungrounded_financing_amount():
    source_text = "公司公告称完成融资，被投资方为甲公司。"
    schema = {"企业融资": ["投资方", "披露时间", "被投资方", "融资金额", "融资轮次", "事件时间", "领投方"]}
    pred_events = [
        {
            "event_type": "企业融资",
            "arguments": [{"role": "被投资方", "argument": "甲公司"}],
        }
    ]

    result = apply_record_corrector(
        pred_events,
        source_text=source_text,
        schema=schema,
        role_alias_map={},
    )

    args = result.events[0]["arguments"]
    assert {"role": "被投资方", "argument": "甲公司"} in args
    assert all(arg["role"] != "融资金额" for arg in args)


def test_apply_structured_event_pipeline_supports_record_corrector_plus_cat_lite():
    source_text = "监管部门约谈甲公司、乙公司。"
    schema = {"被约谈": ["公司名称", "约谈机构"]}
    pred_events = [
        {
            "event_type": "被约谈",
            "arguments": [
                {"role": "公司名称", "argument": "甲公司、乙公司"},
                {"role": "约谈机构", "argument": "监管部门"},
            ],
        }
    ]

    result = apply_structured_event_pipeline(
        pred_events,
        source_text=source_text,
        schema=schema,
        role_alias_map={},
        pipeline_mode="record_corrector+cat_lite",
    )

    assert len(result.events) == 2
    assert result.correction_result is not None
    assert result.cat_result is not None


def test_validate_pipeline_mode_accepts_record_corrector_variants():
    assert validate_pipeline_mode("record_corrector") == "record_corrector"
    assert validate_pipeline_mode("record_corrector+cat_lite") == "record_corrector+cat_lite"
