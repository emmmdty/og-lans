from oglans.data.prompt_builder import ChinesePromptBuilder


def test_system_prompt_includes_schema_constraints():
    schema = {
        "中标": ["中标公司", "中标标的", "中标日期"],
        "亏损": ["公司名称", "净亏损", "财报周期"],
    }
    prompt = ChinesePromptBuilder.build_system_prompt(schema=schema)
    assert "事件类型与合法论元角色" in prompt
    assert "中标公司" in prompt
    assert "净亏损" in prompt


def test_inference_messages_use_schema_prompt():
    schema = {"中标": ["中标公司", "中标标的"]}
    messages = ChinesePromptBuilder.get_messages_for_inference("测试文本", schema=schema)
    assert isinstance(messages, list) and len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "中标公司" in messages[0]["content"]
