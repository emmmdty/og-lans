from oglans.inference.event_probe import apply_event_probe_v2
from oglans.inference.postprocess_profiles import (
    apply_postprocess_profile,
    summarize_postprocess_profile_rows,
)


def test_event_probe_v2_canonicalizes_buyback_price_and_drops_bare_share_fragment():
    source_text = "渣打集团回购65.8万股，每股回购价介乎6.144英镑至6.24英镑。"
    records = [
        {
            "event_type": "股份回购",
            "arguments": [
                {"role": "每股交易价格", "argument": "介乎6.144英镑至6.24英镑"},
                {"role": "每股交易价格", "argument": "股"},
            ],
        }
    ]

    probed, stats = apply_event_probe_v2(records, source_text=source_text)

    assert probed[0]["arguments"] == [{"role": "每股交易价格", "argument": "6.144英镑至6.24英镑"}]
    assert stats["buyback_price_canonicalizations"] == 1
    assert stats["value_fragment_drops"] == 1


def test_event_probe_v2_merges_loss_money_fragments_to_grounded_span():
    source_text = "公司实现归属于挂牌公司股东的净利润-3,648,520.86元。"
    records = [
        {
            "event_type": "亏损",
            "arguments": [
                {"role": "净亏损", "argument": "3"},
                {"role": "净亏损", "argument": "520.86元"},
                {"role": "净亏损", "argument": "648"},
            ],
        }
    ]

    probed, stats = apply_event_probe_v2(records, source_text=source_text)

    assert probed[0]["arguments"] == [{"role": "净亏损", "argument": "3,648,520.86元"}]
    assert stats["grounded_span_merges"] == 1


def test_event_probe_v2_drops_rumor_loss_value():
    source_text = "会上，蔚来澄清此前4年亏损57亿美元的谣言，财报显示第二季度公司亏损达32.85亿元。"
    records = [
        {
            "event_type": "亏损",
            "arguments": [
                {"role": "净亏损", "argument": "57亿美元"},
                {"role": "净亏损", "argument": "32.85亿元"},
            ],
        }
    ]

    probed, stats = apply_event_probe_v2(records, source_text=source_text)

    assert probed[0]["arguments"] == [{"role": "净亏损", "argument": "32.85亿元"}]
    assert stats["contextual_argument_drops"] == 1


def test_event_probe_v2_merges_bid_target_fragments_into_project_span():
    source_text = "公司预中标昭通中心城市餐厨垃圾及病死禽畜资源化和无害化处理项目，中标金额5000万元。"
    records = [
        {
            "event_type": "中标",
            "arguments": [
                {"role": "中标标的", "argument": "昭通中心城市餐厨垃圾"},
                {"role": "中标标的", "argument": "病死禽畜资源化"},
                {"role": "中标标的", "argument": "无害化处理项目"},
            ],
        }
    ]

    probed, stats = apply_event_probe_v2(records, source_text=source_text)

    assert probed[0]["arguments"] == [
        {"role": "中标标的", "argument": "昭通中心城市餐厨垃圾及病死禽畜资源化和无害化处理项目"}
    ]
    assert stats["bid_target_merges"] == 1


def test_apply_postprocess_profile_accepts_none_and_event_probe_v2():
    records = [{"event_type": "中标", "arguments": [{"role": "中标标的", "argument": "项目"}]}]

    unchanged, none_stats = apply_postprocess_profile(records, source_text="公司中标项目。", profile="none")
    changed, probe_stats = apply_postprocess_profile(
        [{"event_type": "股份回购", "arguments": [{"role": "每股交易价格", "argument": "股"}]}],
        source_text="公司股份回购，但价格字段只输出了股。",
        profile="event_probe_v2",
    )

    assert unchanged == records
    assert none_stats["profile"] == "none"
    assert none_stats["changed"] is False
    assert changed == [{"event_type": "股份回购", "arguments": []}]
    assert probe_stats["profile"] == "event_probe_v2"
    assert probe_stats["profile_stats"]["value_fragment_drops"] == 1


def test_summarize_postprocess_profile_rows_aggregates_profile_stats():
    summary = summarize_postprocess_profile_rows(
        [
            {
                "postprocess_profile_stats": {
                    "changed": True,
                    "input_records": 2,
                    "output_records": 1,
                    "profile_stats": {"value_fragment_drops": 1},
                }
            },
            {
                "postprocess_profile_stats": {
                    "changed": False,
                    "input_records": 1,
                    "output_records": 1,
                    "profile_stats": {"value_fragment_drops": 2},
                }
            },
        ]
    )

    assert summary["postprocess_profile_changed_samples"] == 1
    assert summary["postprocess_profile_input_records"] == 3
    assert summary["postprocess_profile_output_records"] == 2
    assert summary["postprocess_profile_stats"]["value_fragment_drops"] == 3
