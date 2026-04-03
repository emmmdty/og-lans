from pathlib import Path

import pytest
import yaml

from oglans.config import ConfigManager
from oglans.data.adapter import DuEEFinAdapter
from oglans.utils.json_parser import parse_event_list_strict


def test_config_requires_model_profile(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "project": {"seed": 3407},
                "model": {"base_model": "Qwen/Qwen3-4B-Instruct-2507", "source": "modelscope"},
                "training": {"mode": "preference"},
                "algorithms": {
                    "lans": {"enabled": True},
                    "scv": {"enabled": True},
                },
                "comparison": {
                    "eval_protocol_path": "./configs/eval_protocol.yaml",
                    "role_alias_map_path": "./configs/role_aliases_duee_fin.yaml",
                    "prompt_builder_version": "phase3_mvp_v1",
                    "parser_version": "phase3_mvp_v1",
                    "normalization_version": "phase3_mvp_v1",
                },
                "evaluation": {"mode": "scored"},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model.profile"):
        ConfigManager.load_config(str(config_path))


def test_adapter_bad_json_line_fails_fast(tmp_path):
    data_dir = tmp_path / "DuEE-Fin"
    data_dir.mkdir()
    schema_path = data_dir / "duee_fin_event_schema.json"
    schema_path.write_text(
        '{"event_type":"中标","role_list":[{"role":"中标公司"}]}\n',
        encoding="utf-8",
    )
    (data_dir / "duee_fin_train.json").write_text(
        '{"id":"1","text":"a","event_list":[]}\n'
        '{"id":"2","text":"b","event_list":[]\n',
        encoding="utf-8",
    )
    adapter = DuEEFinAdapter(data_path=str(data_dir), schema_path=str(schema_path))

    with pytest.raises(ValueError, match="JSON"):
        adapter.load_data("train")


def test_parse_event_list_strict_rejects_markdown_wrapped_payload():
    with pytest.raises(ValueError, match="strict JSON"):
        parse_event_list_strict('```json\n[{"event_type":"中标","arguments":[]}]\n```')


def test_evaluate_api_no_longer_auto_prediction_only():
    repo_root = Path(__file__).resolve().parents[1]
    evaluate_api_text = (repo_root / "evaluate_api.py").read_text(encoding="utf-8")

    assert "进入 prediction-only 模式" not in evaluate_api_text
    assert "evaluation.mode" in evaluate_api_text


def test_prompt_builder_contract_requires_plain_json_output():
    repo_root = Path(__file__).resolve().parents[1]
    prompt_builder_text = (repo_root / "src" / "oglans" / "data" / "prompt_builder.py").read_text(encoding="utf-8")

    assert "<thought>" not in prompt_builder_text
    assert "```json" not in prompt_builder_text
    assert "严格输出 JSON 数组" in prompt_builder_text
