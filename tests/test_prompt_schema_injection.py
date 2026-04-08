from oglans.data.prompt_builder import (
    ChinesePromptBuilder,
    PROMPT_BUILDER_VERSION,
    build_inference_prompt_payload,
    resolve_prompt_settings,
    validate_prompt_variant,
)


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
    assert PROMPT_BUILDER_VERSION == "phase3_mvp_v1"


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
