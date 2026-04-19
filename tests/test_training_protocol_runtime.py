import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "oglans"
    / "utils"
    / "training_protocol.py"
)
spec = importlib.util.spec_from_file_location("training_protocol", str(MODULE_PATH))
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load training_protocol from {MODULE_PATH}")
training_protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training_protocol)


class StubTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        chunks = [f"{item['role']}::{item['content']}" for item in messages]
        if add_generation_prompt:
            chunks.append("assistant::")
        return "\n".join(chunks)


def _build_sample(sample_id: str, text: str, events: list[dict]):
    return SimpleNamespace(
        id=sample_id,
        text=text,
        chosen=json.dumps(events, ensure_ascii=False, indent=2),
        event_types=[event["event_type"] for event in events if event.get("event_type")],
        events=events,
    )


def test_select_training_fit_samples_can_use_frozen_manifest(tmp_path: Path):
    samples = [
        _build_sample("a", "甲公司完成收购。", [{"event_type": "企业收购", "trigger": "收购", "arguments": []}]),
        _build_sample("b", "乙公司中标项目。", [{"event_type": "中标", "trigger": "中标", "arguments": []}]),
        _build_sample("c", "丙公司完成回购。", [{"event_type": "股份回购", "trigger": "回购", "arguments": []}]),
    ]
    manifest_path = tmp_path / "research_split.json"
    manifest_path.write_text(
        json.dumps(
            {
                "seed": 3407,
                "tune_ratio": 0.1,
                "fit_ids": ["a", "c"],
                "tune_ids": ["b"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    selected, manifest = training_protocol.select_training_fit_samples(
        samples,
        split_manifest=manifest_path,
    )

    assert [sample.id for sample in selected] == ["a", "c"]
    assert manifest["manifest_path"] == str(manifest_path)


def test_resolve_trainer_sample_sources_preserves_teacher_silver_after_main_train_fit():
    gold_a = _build_sample(
        "a",
        "甲公司完成收购。",
        [{"event_type": "企业收购", "trigger": "收购", "arguments": []}],
    )
    gold_b = _build_sample(
        "b",
        "乙公司中标项目。",
        [{"event_type": "中标", "trigger": "中标", "arguments": []}],
    )
    silver_a = _build_sample(
        "teacher::a",
        "甲公司完成收购。",
        [{"event_type": "企业收购", "trigger": "收购", "arguments": []}],
    )

    resolved = training_protocol.resolve_trainer_sample_sources(
        [gold_a, silver_a],
        gold_source_samples=[gold_a, gold_b],
        tune_ratio=0.1,
        seed=3407,
        split_manifest={
            "seed": 3407,
            "tune_ratio": 0.1,
            "fit_ids": ["a"],
            "tune_ids": ["b"],
        },
    )

    assert [sample.id for sample in resolved["training_samples"]] == ["a", "teacher::a"]
    assert [sample.id for sample in resolved["fewshot_source_samples"]] == ["a", "b"]
    assert resolved["input_samples_already_selected"] is True
    assert resolved["split_manifest"]["fit_ids"] == ["a"]


def test_expand_training_samples_two_stage_emits_stage1_and_stage2_records():
    schema = {
        "企业收购": ["收购方", "被收购方", "披露时间", "收购完成时间"],
        "中标": ["中标公司", "中标金额", "披露日期"],
    }
    sample_a = _build_sample(
        "a",
        "甲公司于2024年4月完成对乙公司的收购。",
        [
            {
                "event_type": "企业收购",
                "trigger": "收购",
                "arguments": [
                    {"role": "收购方", "argument": "甲公司"},
                    {"role": "被收购方", "argument": "乙公司"},
                    {"role": "收购完成时间", "argument": "2024年4月"},
                ],
            }
        ],
    )
    sample_b = _build_sample(
        "b",
        "丙公司于2024年5月完成对丁公司的收购。",
        [
            {
                "event_type": "企业收购",
                "trigger": "收购",
                "arguments": [
                    {"role": "收购方", "argument": "丙公司"},
                    {"role": "被收购方", "argument": "丁公司"},
                    {"role": "收购完成时间", "argument": "2024年5月"},
                ],
            }
        ],
    )
    example_pool = [
        {
            "id": "train_fit:a",
            "user": "user-a",
            "assistant": "assistant-a",
            "source_text": sample_a.text,
            "event_types": ["企业收购"],
            "triggers": ["收购"],
            "roles": ["收购方", "被收购方", "收购完成时间"],
            "keywords": ["收购", "完成"],
        },
        {
            "id": "train_fit:b",
            "user": "user-b",
            "assistant": "assistant-b",
            "source_text": sample_b.text,
            "event_types": ["企业收购"],
            "triggers": ["收购"],
            "roles": ["收购方", "被收购方", "收购完成时间"],
            "keywords": ["收购", "完成"],
        },
    ]

    expanded = training_protocol.expand_training_samples(
        [sample_a],
        tokenizer=StubTokenizer(),
        schema=schema,
        stage_mode="two_stage",
        prompt_variant="fewshot",
        fewshot_num_examples=1,
        fewshot_selection_mode="dynamic",
        fewshot_example_pool=example_pool,
    )

    assert [item["training_stage"] for item in expanded] == [
        "stage1_event_type",
        "stage2_extraction",
    ]
    stage1_record, stage2_record = expanded
    assert stage1_record["lans_eligible"] is False
    assert stage1_record["use_precomputed_rejected"] is True
    assert json.loads(stage1_record["chosen"]) == [{"event_type": "企业收购", "arguments": []}]
    assert json.loads(stage1_record["rejected"]) != json.loads(stage1_record["chosen"])
    assert stage2_record["lans_eligible"] is True
    assert stage2_record["stage2_schema_event_types"] == ["企业收购"]
    assert stage2_record["fewshot_example_ids"] == ["train_fit:b"]


def test_expand_training_samples_two_stage_per_type_emits_one_stage2_record_per_event_type():
    schema = {
        "企业收购": ["收购方", "被收购方", "披露时间", "收购完成时间"],
        "中标": ["中标公司", "中标金额", "披露日期"],
    }
    sample = _build_sample(
        "typed-a",
        "甲公司于2024年4月完成对乙公司的收购，并中标智慧园区项目。",
        [
            {
                "event_type": "企业收购",
                "trigger": "收购",
                "arguments": [
                    {"role": "收购方", "argument": "甲公司"},
                    {"role": "被收购方", "argument": "乙公司"},
                    {"role": "收购完成时间", "argument": "2024年4月"},
                ],
            },
            {
                "event_type": "中标",
                "trigger": "中标",
                "arguments": [
                    {"role": "中标公司", "argument": "甲公司"},
                    {"role": "中标标的", "argument": "智慧园区项目"},
                ],
            },
        ],
    )

    expanded = training_protocol.expand_training_samples(
        [sample],
        tokenizer=StubTokenizer(),
        schema=schema,
        stage_mode="two_stage_per_type",
        prompt_variant="zeroshot",
        fewshot_num_examples=0,
        fewshot_selection_mode="dynamic",
        fewshot_example_pool=[],
    )

    assert [item["training_stage"] for item in expanded] == [
        "stage1_event_type",
        "stage2_extraction",
        "stage2_extraction",
    ]
    assert [item["stage_mode"] for item in expanded] == [
        "two_stage_per_type",
        "two_stage_per_type",
        "two_stage_per_type",
    ]
    assert expanded[1]["target_event_types"] == ["企业收购"]
    assert expanded[1]["stage2_schema_event_types"] == ["企业收购"]
    assert {event["event_type"] for event in json.loads(expanded[1]["chosen"])} == {"企业收购"}
    assert expanded[2]["target_event_types"] == ["中标"]
    assert expanded[2]["stage2_schema_event_types"] == ["中标"]
    assert {event["event_type"] for event in json.loads(expanded[2]["chosen"])} == {"中标"}


def test_resolve_training_resume_settings_distinguishes_warm_start_and_resume():
    warm_start = training_protocol.resolve_training_resume_settings(
        {
            "warm_start_from_checkpoint": "/tmp/warm-start",
        }
    )
    resume_training = training_protocol.resolve_training_resume_settings(
        {
            "resume_training_from": "/tmp/resume-training",
        }
    )

    assert warm_start["mode"] == "warm_start"
    assert warm_start["checkpoint_path"] == "/tmp/warm-start"
    assert warm_start["restores_training_state"] is False
    assert resume_training["mode"] == "resume_training"
    assert resume_training["checkpoint_path"] == "/tmp/resume-training"
    assert resume_training["restores_training_state"] is True


def test_resolve_training_resume_settings_rejects_legacy_and_conflicting_fields():
    with pytest.raises(ValueError, match="resume_from_checkpoint"):
        training_protocol.resolve_training_resume_settings(
            {
                "resume_from_checkpoint": "/tmp/legacy",
            }
        )

    with pytest.raises(ValueError, match="mutually exclusive"):
        training_protocol.resolve_training_resume_settings(
            {
                "warm_start_from_checkpoint": "/tmp/warm-start",
                "resume_training_from": "/tmp/resume-training",
            }
        )


def test_expand_training_samples_single_pass_excludes_self_from_fewshot_pool():
    schema = {"中标": ["中标公司", "中标金额", "披露日期"]}
    sample_a = _build_sample(
        "a",
        "甲公司于2024年1月中标智慧园区项目。",
        [
            {
                "event_type": "中标",
                "trigger": "中标",
                "arguments": [
                    {"role": "中标公司", "argument": "甲公司"},
                    {"role": "中标标的", "argument": "智慧园区项目"},
                    {"role": "披露日期", "argument": "2024年1月"},
                ],
            }
        ],
    )
    sample_b = _build_sample(
        "b",
        "乙公司于2024年2月中标算力中心项目。",
        [
            {
                "event_type": "中标",
                "trigger": "中标",
                "arguments": [
                    {"role": "中标公司", "argument": "乙公司"},
                    {"role": "中标标的", "argument": "算力中心项目"},
                    {"role": "披露日期", "argument": "2024年2月"},
                ],
            }
        ],
    )
    example_pool = [
        {
            "id": "train_fit:a",
            "user": "user-a",
            "assistant": "assistant-a",
            "source_text": sample_a.text,
            "event_types": ["中标"],
            "triggers": ["中标"],
            "roles": ["中标公司", "披露日期"],
            "keywords": ["中标", "项目"],
        },
        {
            "id": "train_fit:b",
            "user": "user-b",
            "assistant": "assistant-b",
            "source_text": sample_b.text,
            "event_types": ["中标"],
            "triggers": ["中标"],
            "roles": ["中标公司", "披露日期"],
            "keywords": ["中标", "项目"],
        },
    ]

    expanded = training_protocol.expand_training_samples(
        [sample_a],
        tokenizer=StubTokenizer(),
        schema=schema,
        stage_mode="single_pass",
        prompt_variant="fewshot",
        fewshot_num_examples=1,
        fewshot_selection_mode="dynamic",
        fewshot_example_pool=example_pool,
    )

    assert len(expanded) == 1
    assert expanded[0]["fewshot_example_ids"] == ["train_fit:b"]


def test_training_cache_metadata_matches_requires_protocol_fields():
    expected = training_protocol.build_training_cache_metadata(
        dataset_name="DuEE-Fin",
        training_mode="preference",
        stage_mode="two_stage",
        prompt_variant="fewshot",
        fewshot_num_examples=3,
        fewshot_selection_mode="dynamic",
        fewshot_pool_split="train_fit",
        research_split_manifest_hash="abc123",
        prompt_builder_version="phase3_mvp_v1",
        parser_version="phase3_mvp_v1",
        normalization_version="phase3_mvp_v1",
        model_profile="qwen3_instruct",
        max_seq_length=3072,
        use_lans=True,
        taxonomy_hash="schema123",
    )

    assert training_protocol.training_cache_metadata_matches(expected, dict(expected)) is True
    mutated = dict(expected)
    mutated["stage_mode"] = "single_pass"
    assert training_protocol.training_cache_metadata_matches(expected, mutated) is False
