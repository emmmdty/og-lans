import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "evaluate_api.py"
spec = importlib.util.spec_from_file_location("evaluate_api", str(SCRIPT_PATH))
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load module from {SCRIPT_PATH}")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_resolve_api_runtime_config_cli_base_url_overrides_env_and_config(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://env.deepseek.example/v1")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.openai.example/v1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")

    resolved = mod.resolve_api_runtime_config(
        base_url_override="https://cli.example/v1",
        model_override="cli-model",
        api_cfg={
            "base_url": "https://config.example/v1",
            "model": "config-model",
            "api_key": None,
        },
        environ=os.environ,
    )

    assert resolved["base_url"] == "https://cli.example/v1"
    assert resolved["base_url_source"] == "cli"
    assert resolved["model_name"] == "cli-model"
    assert resolved["model_source"] == "cli"
    assert resolved["api_key"] == "deepseek-secret"
    assert resolved["api_key_source"] == "env:DEEPSEEK_API_KEY"


def test_resolve_api_runtime_config_env_base_url_overrides_config(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://env.openai.example/v1")
    monkeypatch.delenv("DEEPSEEK_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_DEFAULT_MODEL", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

    resolved = mod.resolve_api_runtime_config(
        base_url_override=None,
        model_override=None,
        api_cfg={
            "base_url": "https://config.example/v1",
            "model": "config-model",
            "api_key": None,
        },
        environ=os.environ,
    )

    assert resolved["base_url"] == "https://env.openai.example/v1"
    assert resolved["base_url_source"] == "env:OPENAI_BASE_URL"
    assert resolved["model_name"] == "config-model"
    assert resolved["model_source"] == "config"
    assert resolved["api_key"] == "openai-secret"
    assert resolved["api_key_source"] == "env:OPENAI_API_KEY"


def test_resolve_api_runtime_config_falls_back_to_config(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("DEEPSEEK_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    resolved = mod.resolve_api_runtime_config(
        base_url_override=None,
        model_override=None,
        api_cfg={
            "base_url": "https://config.example/v1",
            "model": "config-model",
            "api_key": "config-secret",
        },
        environ=os.environ,
    )

    assert resolved["base_url"] == "https://config.example/v1"
    assert resolved["base_url_source"] == "config"
    assert resolved["model_name"] == "config-model"
    assert resolved["model_source"] == "config"
    assert resolved["api_key"] == "config-secret"
    assert resolved["api_key_source"] == "config"


def test_resolve_api_runtime_config_prefers_provider_default_model_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-reasoner")
    monkeypatch.setenv("OPENAI_DEFAULT_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")

    resolved = mod.resolve_api_runtime_config(
        base_url_override=None,
        model_override=None,
        api_cfg={
            "base_url": None,
            "model": "config-model",
            "api_key": None,
        },
        environ=os.environ,
    )

    assert resolved["base_url"] == "https://api.deepseek.com"
    assert resolved["base_url_source"] == "env:DEEPSEEK_BASE_URL"
    assert resolved["model_name"] == "deepseek-reasoner"
    assert resolved["model_source"] == "env:DEEPSEEK_DEFAULT_MODEL"


def test_resolve_api_runtime_config_prefers_openai_default_model_for_openai_endpoint(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.openai.example/v1")
    monkeypatch.setenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat")
    monkeypatch.setenv("OPENAI_DEFAULT_MODEL", "gpt-4.1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

    resolved = mod.resolve_api_runtime_config(
        base_url_override=None,
        model_override=None,
        api_cfg={
            "base_url": None,
            "model": "config-model",
            "api_key": None,
        },
        environ=os.environ,
    )

    assert resolved["base_url"] == "https://provider.openai.example/v1"
    assert resolved["base_url_source"] == "env:OPENAI_BASE_URL"
    assert resolved["model_name"] == "gpt-4.1"
    assert resolved["model_source"] == "env:OPENAI_DEFAULT_MODEL"


def test_build_api_fewshot_example_pool_extracts_sample_metadata():
    class DummySample:
        def __init__(self, sample_id, text, event_types, events):
            self.id = sample_id
            self.text = text
            self.event_types = event_types
            self.events = events

    schema = {"中标": ["中标公司", "中标金额"]}
    sample = DummySample(
        "s1",
        "华建科技于2024年8月1日中标智慧园区项目。",
        ["中标"],
        [
            {
                "event_type": "中标",
                "trigger": "中标",
                "arguments": [
                    {"role": "中标公司", "argument": "华建科技"},
                    {"role": "中标金额", "argument": "1.2亿元"},
                ],
            }
        ],
    )

    pool = mod.build_api_fewshot_example_pool([sample], schema=schema, source_split="train")

    assert len(pool) == 1
    example = pool[0]
    assert example["id"] == "train:s1"
    assert example["event_types"] == ["中标"]
    assert example["triggers"] == ["中标"]
    assert "中标公司" in example["roles"]
    assert "【文本内容】" in example["user"]


def test_resolve_stage_settings_prefers_cli_overrides():
    resolved = mod.resolve_stage_settings(
        stage_mode="two_stage_per_type",
        fewshot_selection_mode="contrastive",
        fewshot_pool_split="train_fit",
        comparison_cfg={"stage_mode": "single_pass"},
    )

    assert resolved["stage_mode"] == "two_stage_per_type"
    assert resolved["fewshot_selection_mode"] == "contrastive"
    assert resolved["fewshot_pool_split"] == "train_fit"


def test_compute_sample_metric_row_records_gold_event_count_for_multiplicity_ci():
    evaluator = mod.AcademicEventEvaluator()
    gold = [
        {
            "event_type": "企业收购",
            "arguments": [{"role": "收购方", "argument": "甲公司"}],
        },
        {
            "event_type": "企业融资",
            "arguments": [{"role": "融资金额", "argument": "10亿元"}],
        },
    ]

    row = mod.compute_sample_metric_row(evaluator, pred_events=gold[:1], gold_events=gold)

    assert row["gold_event_count"] == 2


def test_process_single_sample_two_stage_routes_stage1_event_types_into_fewshot_selection(monkeypatch):
    captured = {}

    def fake_perform_api_inference(client, model, messages, max_retries, json_mode):
        if len(messages) == 2 and "请识别以下金融文本中出现的事件类型" in messages[1]["content"]:
            return (
                '[{"event_type":"中标","arguments":[]}]',
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                True,
                None,
                {},
            )
        return (
            "[]",
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            True,
            None,
            {},
        )

    def fake_parse_event_list_strict_with_diagnostics(text):
        try:
            return json.loads(text), {"success": True, "error": None}
        except json.JSONDecodeError:
            return [], {"success": False, "error": "json_error"}

    def fake_build_inference_prompt_payload(**kwargs):
        captured.update(kwargs)
        return {
            "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            "fewshot_selection_mode": kwargs.get("fewshot_selection_mode", "none"),
            "fewshot_example_ids": [],
            "fewshot_count": 0,
        }

    monkeypatch.setattr(mod, "perform_api_inference", fake_perform_api_inference)
    monkeypatch.setattr(mod, "parse_event_list_strict_with_diagnostics", fake_parse_event_list_strict_with_diagnostics)
    monkeypatch.setattr(mod, "build_inference_prompt_payload", fake_build_inference_prompt_payload)

    sample = SimpleNamespace(
        id="s1",
        text="华建科技公告称公司中标智慧园区升级项目。",
        events=[],
    )

    result = mod.process_single_sample(
        client=object(),
        model="deepseek-chat",
        max_retries=1,
        sample_idx=0,
        sample=sample,
        use_fewshot=True,
        fewshot_num_examples=2,
        fewshot_selection_mode="dynamic",
        fewshot_example_pool=[],
        json_mode="auto",
        schema={"中标": ["中标公司", "中标标的"]},
        pipeline_mode="e2e",
        stage_mode="two_stage",
        role_alias_map=None,
    )

    assert captured["target_event_types"] == ["中标"]
    assert result["stage_meta"]["stage1_predicted_event_types"] == ["中标"]


def test_process_single_sample_applies_postprocess_profile_after_pipeline(monkeypatch):
    def fake_perform_api_inference(client, model, messages, max_retries, json_mode):
        return (
            '[{"event_type":"股份回购","arguments":[{"role":"每股交易价格","argument":"股"}]}]',
            {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            True,
            None,
            {},
        )

    def fake_parse_event_list_strict_with_diagnostics(text):
        return json.loads(text), {"success": True, "error": None}

    def fake_pipeline(pred_events, **kwargs):
        return SimpleNamespace(events=pred_events, cat_result=None, correction_result=None)

    monkeypatch.setattr(mod, "perform_api_inference", fake_perform_api_inference)
    monkeypatch.setattr(mod, "parse_event_list_strict_with_diagnostics", fake_parse_event_list_strict_with_diagnostics)
    monkeypatch.setattr(mod, "apply_structured_event_pipeline", fake_pipeline)

    sample = SimpleNamespace(id="s-post", text="公司回购股份。", events=[])
    result = mod.process_single_sample(
        client=object(),
        model="deepseek-chat",
        max_retries=1,
        sample_idx=0,
        sample=sample,
        use_fewshot=False,
        fewshot_num_examples=0,
        fewshot_selection_mode="static",
        fewshot_example_pool=None,
        json_mode="auto",
        schema={"股份回购": ["每股交易价格"]},
        pipeline_mode="e2e",
        postprocess_profile="event_probe_v2",
        stage_mode="single_pass",
        role_alias_map=None,
    )

    assert result["pred_events"] == [{"event_type": "股份回购", "arguments": []}]
    assert result["postprocess_profile_stats"]["profile"] == "event_probe_v2"
    assert result["postprocess_profile_stats"]["profile_stats"]["value_fragment_drops"] == 1


def test_process_single_sample_two_stage_per_type_extracts_each_event_type_separately(monkeypatch):
    build_calls = []

    def fake_perform_api_inference(client, model, messages, max_retries, json_mode):
        user_content = messages[1]["content"]
        if "请识别以下金融文本中出现的事件类型" in user_content:
            return (
                '[{"event_type":"中标","arguments":[]},{"event_type":"股份回购","arguments":[]}]',
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                True,
                None,
                {},
            )
        if "target=中标" in user_content:
            return (
                '[{"event_type":"中标","arguments":[{"role":"中标公司","argument":"华建科技"}]}]',
                {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                True,
                None,
                {},
            )
        if "target=股份回购" in user_content:
            return (
                '[{"event_type":"股份回购","arguments":[{"role":"回购方","argument":"华建科技"}]}]',
                {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                True,
                None,
                {},
            )
        raise AssertionError(f"unexpected messages: {messages}")

    def fake_parse_event_list_strict_with_diagnostics(text):
        try:
            return json.loads(text), {"success": True, "error": None}
        except json.JSONDecodeError:
            return [], {"success": False, "error": "json_error"}

    def fake_build_inference_prompt_payload(**kwargs):
        build_calls.append(kwargs)
        target = (kwargs.get("target_event_types") or ["all"])[0]
        return {
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": f"target={target};text={kwargs.get('text', '')}"},
            ],
            "fewshot_selection_mode": kwargs.get("fewshot_selection_mode", "none"),
            "fewshot_example_ids": [f"ex-{target}"],
            "fewshot_count": 1,
            "fewshot_target_event_types": kwargs.get("target_event_types", []),
            "fewshot_contrastive_warnings": [],
        }

    monkeypatch.setattr(mod, "perform_api_inference", fake_perform_api_inference)
    monkeypatch.setattr(mod, "parse_event_list_strict_with_diagnostics", fake_parse_event_list_strict_with_diagnostics)
    monkeypatch.setattr(mod, "build_inference_prompt_payload", fake_build_inference_prompt_payload)

    sample = SimpleNamespace(
        id="s2",
        text="华建科技公告称公司中标智慧园区升级项目，并实施股份回购。",
        events=[],
    )

    result = mod.process_single_sample(
        client=object(),
        model="deepseek-chat",
        max_retries=1,
        sample_idx=0,
        sample=sample,
        use_fewshot=True,
        fewshot_num_examples=2,
        fewshot_selection_mode="contrastive",
        fewshot_example_pool=[],
        json_mode="auto",
        schema={"中标": ["中标公司"], "股份回购": ["回购方"]},
        pipeline_mode="e2e",
        stage_mode="two_stage_per_type",
        role_alias_map=None,
    )

    assert [call["target_event_types"] for call in build_calls] == [["中标"], ["股份回购"]]
    assert {event["event_type"] for event in result["pred_events"]} == {"中标", "股份回购"}
    assert result["stage_meta"]["typed_stage2_call_count"] == 2
    assert result["prompt_meta"]["fewshot_example_ids"] == ["ex-中标", "ex-股份回购"]


def test_process_single_sample_two_stage_per_type_surfaces_partial_stage_failures(monkeypatch):
    def fake_perform_api_inference(client, model, messages, max_retries, json_mode):
        user_content = messages[1]["content"]
        if "请识别以下金融文本中出现的事件类型" in user_content:
            return (
                '[{"event_type":"中标","arguments":[]},{"event_type":"股份回购","arguments":[]}]',
                {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                True,
                None,
                {"request_id": "stage1"},
            )
        if "target=中标" in user_content:
            return (
                "[]",
                {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                False,
                "timeout",
                {"request_id": "typed-zhongbiao"},
            )
        if "target=股份回购" in user_content:
            return (
                '[{"event_type":"股份回购","arguments":[{"role":"回购方","argument":"华建科技"}]}]',
                {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
                True,
                None,
                {"request_id": "typed-huigou"},
            )
        raise AssertionError(f"unexpected messages: {messages}")

    def fake_parse_event_list_strict_with_diagnostics(text):
        return json.loads(text), {"success": True, "error": None}

    def fake_build_inference_prompt_payload(**kwargs):
        target = (kwargs.get("target_event_types") or ["all"])[0]
        return {
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": f"target={target};text={kwargs.get('text', '')}"},
            ],
            "fewshot_selection_mode": kwargs.get("fewshot_selection_mode", "none"),
            "fewshot_example_ids": [f"ex-{target}"],
            "fewshot_count": 1,
            "fewshot_target_event_types": kwargs.get("target_event_types", []),
            "fewshot_contrastive_warnings": [],
        }

    monkeypatch.setattr(mod, "perform_api_inference", fake_perform_api_inference)
    monkeypatch.setattr(mod, "parse_event_list_strict_with_diagnostics", fake_parse_event_list_strict_with_diagnostics)
    monkeypatch.setattr(mod, "build_inference_prompt_payload", fake_build_inference_prompt_payload)

    sample = SimpleNamespace(
        id="s3",
        text="华建科技公告称公司中标智慧园区升级项目，并实施股份回购。",
        events=[],
    )

    result = mod.process_single_sample(
        client=object(),
        model="deepseek-chat",
        max_retries=1,
        sample_idx=0,
        sample=sample,
        use_fewshot=True,
        fewshot_num_examples=2,
        fewshot_selection_mode="contrastive",
        fewshot_example_pool=[],
        json_mode="auto",
        schema={"中标": ["中标公司"], "股份回购": ["回购方"]},
        pipeline_mode="e2e",
        stage_mode="two_stage_per_type",
        role_alias_map=None,
    )

    assert result["api_success"] is False
    assert result["api_error"]["stage2"][0]["event_type"] == "中标"
    assert result["api_error"]["stage2"][0]["api_error"] == "timeout"
    assert result["response_meta"]["stage2"][0]["response_text"] == "[]"
