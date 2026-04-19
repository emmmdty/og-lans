from oglans.data.prompt_builder import (
    ChinesePromptBuilder,
    PROMPT_BUILDER_VERSION,
    build_inference_prompt_payload,
    resolve_prompt_settings,
    validate_prompt_variant,
)
import json


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        rendered = [f"{item['role']}::{item['content']}" for item in messages]
        if add_generation_prompt:
            rendered.append("assistant::")
        return "\n".join(rendered)


def test_system_prompt_includes_schema_constraints():
    schema = {
        "中标": ["中标公司", "中标标的", "中标日期"],
        "亏损": ["公司名称", "净亏损", "财报周期"],
    }
    prompt = ChinesePromptBuilder.build_system_prompt(schema=schema)
    assert "事件类型与合法论元角色" in prompt
    assert "中标公司" in prompt
    assert "净亏损" in prompt
    assert "使用 schema 中的标准角色名" in prompt
    assert "拆成多个独立的 arguments 项" in prompt
    assert "严格输出 JSON 数组" in prompt


def test_inference_messages_use_schema_prompt():
    schema = {"中标": ["中标公司", "中标标的"]}
    messages = ChinesePromptBuilder.get_messages_for_inference("测试文本", schema=schema)
    assert isinstance(messages, list) and len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "中标公司" in messages[0]["content"]


def test_inference_prompt_payload_keeps_materializations_in_sync():
    schema = {"中标": ["中标公司", "中标标的"]}
    tokenizer = DummyTokenizer()

    payload = build_inference_prompt_payload(
        "测试文本",
        tokenizer=tokenizer,
        schema=schema,
        use_oneshot=True,
        num_examples=2,
    )

    expected_messages = ChinesePromptBuilder.get_messages_with_oneshot(
        "测试文本",
        num_examples=2,
        schema=schema,
    )

    assert payload["messages"] == expected_messages
    assert payload["prompt_variant"] == "fewshot"
    assert payload["fewshot_count"] == 2
    assert payload["schema_enabled"] is True
    assert payload["formatted_text"].startswith("system::")
    assert "assistant::" in payload["formatted_text"]
    assert PROMPT_BUILDER_VERSION == "phase3_mvp_v2"


def test_resolve_prompt_settings_prefers_explicit_prompt_variant():
    settings = resolve_prompt_settings(
        prompt_variant="fewshot",
        fewshot_num_examples=2,
        use_oneshot=None,
        default_prompt_variant="zeroshot",
        default_num_examples=3,
    )

    assert settings["prompt_variant"] == "fewshot"
    assert settings["use_oneshot"] is True
    assert settings["fewshot_num_examples"] == 2


def test_resolve_prompt_settings_supports_legacy_use_oneshot_alias():
    settings = resolve_prompt_settings(
        prompt_variant=None,
        fewshot_num_examples=None,
        use_oneshot=True,
        default_prompt_variant="zeroshot",
        default_num_examples=3,
    )

    assert settings["prompt_variant"] == "fewshot"
    assert settings["fewshot_num_examples"] == 3


def test_validate_prompt_variant_rejects_unknown_value():
    try:
        validate_prompt_variant("oneshot")
    except ValueError as exc:
        assert "Unsupported prompt_variant" in str(exc)
    else:
        raise AssertionError("validate_prompt_variant should reject unknown variants")


def test_system_prompt_warns_against_cross_event_role_generalization():
    prompt = ChinesePromptBuilder.build_system_prompt()

    assert "时间相关 role 也必须与 schema 完全一致" in prompt
    assert "证券代码仅适用于公司上市事件" in prompt
    assert "股份回购应使用回购完成时间，不要使用事件时间" in prompt
    assert "中标应使用披露日期，不要使用披露时间" in prompt


def test_fewshot_examples_use_only_schema_valid_roles_for_demonstrated_events():
    valid_roles = {
        "被约谈": {"公司名称", "披露时间", "被约谈时间", "约谈机构"},
        "股份回购": {"回购方", "披露时间", "回购股份数量", "每股交易价格", "占公司总股本比例", "交易金额", "回购完成时间"},
        "中标": {"中标公司", "中标标的", "中标金额", "招标方", "中标日期", "披露日期"},
    }

    for example in ChinesePromptBuilder.FEW_SHOT_EXAMPLES:
        events = json.loads(example["assistant"])
        for event in events:
            assert event["event_type"] in valid_roles
            for argument in event["arguments"]:
                assert argument["role"] in valid_roles[event["event_type"]]


def test_fewshot_examples_include_same_type_multi_record_split_demonstration():
    examples = ChinesePromptBuilder.select_fewshot_examples(num_examples=3)
    found_split_demo = False

    for example in examples:
        events = json.loads(example["assistant"])
        event_types = [event["event_type"] for event in events]
        if any(event_types.count(event_type) > 1 for event_type in set(event_types)):
            found_split_demo = True
            break

    assert found_split_demo is True


def test_select_fewshot_examples_dynamic_prefers_matching_event_examples():
    example_pool = [
        {
            "id": "pledge-example",
            "user": "u1",
            "assistant": "a1",
            "source_text": "控股股东将其股份质押给银行。",
            "event_types": ["质押"],
            "triggers": ["质押"],
            "keywords": ["质押", "股东", "银行"],
        },
        {
            "id": "bid-example",
            "user": "u2",
            "assistant": "a2",
            "source_text": "公司中标智慧园区项目。",
            "event_types": ["中标"],
            "triggers": ["中标"],
            "keywords": ["中标", "项目", "招标方"],
        },
        {
            "id": "buyback-example",
            "user": "u3",
            "assistant": "a3",
            "source_text": "公司实施股份回购计划。",
            "event_types": ["股份回购"],
            "triggers": ["回购"],
            "keywords": ["回购", "股份", "交易金额"],
        },
    ]

    selected = ChinesePromptBuilder.select_fewshot_examples(
        num_examples=2,
        text="公告称控股股东已将所持股份质押给银行并办理质押登记。",
        selection_mode="dynamic",
        example_pool=example_pool,
    )

    assert [item["id"] for item in selected][:1] == ["pledge-example"]
    assert "bid-example" not in [item["id"] for item in selected]


def test_build_inference_prompt_payload_records_dynamic_fewshot_example_ids():
    example_pool = [
        {
            "id": "bid-example",
            "user": "示例用户",
            "assistant": "[]",
            "source_text": "公司中标智慧园区项目。",
            "event_types": ["中标"],
            "triggers": ["中标"],
            "keywords": ["中标", "项目"],
        },
        {
            "id": "buyback-example",
            "user": "示例用户2",
            "assistant": "[]",
            "source_text": "公司实施股份回购计划。",
            "event_types": ["股份回购"],
            "triggers": ["回购"],
            "keywords": ["回购", "股份"],
        },
    ]

    payload = build_inference_prompt_payload(
        "华建科技公告称公司中标智慧园区升级项目。",
        prompt_variant="fewshot",
        num_examples=1,
        fewshot_selection_mode="dynamic",
        fewshot_example_pool=example_pool,
    )

    assert payload["fewshot_count"] == 1
    assert payload["fewshot_selection_mode"] == "dynamic"
    assert payload["fewshot_example_ids"] == ["bid-example"]


def test_dynamic_selection_reranks_acquisition_queries_toward_focused_examples():
    example_pool = [
        {
            "id": "mixed-acquisition-example",
            "user": "u1",
            "assistant": "a1",
            "source_text": "国家电网公司签署股权购买协议推进并购贷款安排，拟全资收购目标公司100%股权并以相关股权质押给银行。",
            "event_types": ["企业收购", "质押"],
            "triggers": ["收购", "质押"],
            "keywords": ["国家电网公司", "签署", "股权", "购买", "协议", "并购", "贷款", "收购", "100%股权", "质押"],
        },
        {
            "id": "focused-acquisition-example",
            "user": "u2",
            "assistant": "a2",
            "source_text": "公司收购目标公司。",
            "event_types": ["企业收购"],
            "triggers": ["收购"],
            "keywords": ["收购"],
        },
    ]

    selected = ChinesePromptBuilder.select_fewshot_examples(
        num_examples=1,
        text="国家电网公司与卖方签署股权购买协议，拟全资收购目标公司100%股权并完成交割。",
        selection_mode="dynamic",
        example_pool=example_pool,
    )

    assert selected[0]["id"] == "focused-acquisition-example"


def test_select_fewshot_examples_dynamic_supports_explicit_target_event_types():
    example_pool = [
        {
            "id": "bid-example",
            "user": "u1",
            "assistant": "a1",
            "source_text": "公司中标智慧园区项目。",
            "event_types": ["中标"],
            "triggers": ["中标"],
            "keywords": ["中标", "项目", "招标方"],
        },
        {
            "id": "buyback-example",
            "user": "u2",
            "assistant": "a2",
            "source_text": "公司实施股份回购计划。",
            "event_types": ["股份回购"],
            "triggers": ["回购"],
            "keywords": ["回购", "股份", "交易金额"],
        },
    ]

    selected = ChinesePromptBuilder.select_fewshot_examples(
        num_examples=1,
        text="董事会审议并发布专项方案。",
        selection_mode="dynamic",
        example_pool=example_pool,
        target_event_types=["中标"],
    )

    assert [item["id"] for item in selected] == ["bid-example"]


def test_build_inference_prompt_payload_contrastive_appends_confusion_guidance():
    example_pool = [
        {
            "id": "bid-example-1",
            "user": "示例用户1",
            "assistant": "[]",
            "source_text": "华建科技公告称公司中标智慧园区项目。",
            "event_types": ["中标"],
            "triggers": ["中标"],
            "keywords": ["中标", "项目", "招标方"],
        },
        {
            "id": "bid-example-2",
            "user": "示例用户2",
            "assistant": "[]",
            "source_text": "另一家公司中标数字基建项目。",
            "event_types": ["中标"],
            "triggers": ["中标"],
            "keywords": ["中标", "项目", "金额"],
        },
    ]

    payload = build_inference_prompt_payload(
        "华建科技公告称公司中标智慧园区升级项目。",
        prompt_variant="fewshot",
        num_examples=2,
        fewshot_selection_mode="contrastive",
        fewshot_example_pool=example_pool,
        target_event_types=["中标"],
    )

    assert payload["fewshot_selection_mode"] == "contrastive"
    assert payload["fewshot_example_ids"] == ["bid-example-1", "bid-example-2"]
    assert payload["fewshot_contrastive_warnings"]
    assert any(
        "易混淆提醒" in message["content"] and "中标标的" in message["content"]
        for message in payload["messages"]
    )
