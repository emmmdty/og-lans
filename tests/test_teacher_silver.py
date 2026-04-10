import json
from pathlib import Path

from oglans.utils.teacher_silver import (
    build_teacher_silver_records,
    compute_role_overlap,
    load_duee_fin_text_index,
    load_teacher_silver_samples,
    save_jsonl,
)


def test_compute_role_overlap_uses_event_role_micro_f1():
    primary = [
        {
            "event_type": "企业融资",
            "arguments": [
                {"role": "融资方", "argument": "甲公司"},
                {"role": "融资金额", "argument": "10亿元"},
            ],
        }
    ]
    secondary = [
        {
            "event_type": "企业融资",
            "arguments": [
                {"role": "融资方", "argument": "甲公司"},
                {"role": "投资方", "argument": "乙资本"},
            ],
        }
    ]
    assert compute_role_overlap(primary, secondary) == 0.5


def test_build_teacher_silver_records_requires_consensus_and_dataset_text(tmp_path: Path):
    dataset_file = tmp_path / "duee_fin_train.json"
    dataset_file.write_text(
        json.dumps(
            {
                "id": "train-1",
                "text": "甲公司完成10亿元融资，乙资本领投。",
                "event_list": [],
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    text_index = load_duee_fin_text_index(dataset_file)
    primary_rows = [
        {
            "id": "train-1",
            "parse_success": True,
            "pipeline_mode": "e2e",
            "prediction_canonical": [
                {
                    "event_type": "企业融资",
                    "arguments": [
                        {"role": "融资方", "argument": "甲公司"},
                        {"role": "融资金额", "argument": "10亿元"},
                    ],
                }
            ],
            "text_preview": "甲公司完成10亿元融资...",
        }
    ]
    secondary_rows = [
        {
            "id": "train-1",
            "parse_success": True,
            "pipeline_mode": "record_corrector+cat_lite",
            "pred_canonical": [
                {
                    "event_type": "企业融资",
                    "arguments": [
                        {"role": "融资方", "argument": "甲公司"},
                        {"role": "融资金额", "argument": "10亿元"},
                    ],
                }
            ],
        }
    ]

    records, summary = build_teacher_silver_records(
        primary_rows,
        secondary_rows=secondary_rows,
        text_index=text_index,
        allowed_ids={"train-1"},
        require_consensus=True,
        min_role_overlap=0.8,
    )

    assert summary["kept_count"] == 1
    assert records[0]["text"] == "甲公司完成10亿元融资，乙资本领投。"
    assert records[0]["teacher_meta"]["consensus_mode"] == "dual_source"
    assert records[0]["teacher_meta"]["agreement_score"] == 1.0


def test_build_teacher_silver_records_skips_event_type_mismatch():
    primary_rows = [
        {
            "id": "sample-1",
            "text": "甲公司完成收购。",
            "parse_success": True,
            "prediction_canonical": [
                {"event_type": "企业收购", "arguments": [{"role": "收购方", "argument": "甲公司"}]}
            ],
        }
    ]
    secondary_rows = [
        {
            "id": "sample-1",
            "text": "甲公司完成收购。",
            "parse_success": True,
            "pred_canonical": [
                {"event_type": "企业融资", "arguments": [{"role": "融资方", "argument": "甲公司"}]}
            ],
        }
    ]

    records, summary = build_teacher_silver_records(
        primary_rows,
        secondary_rows=secondary_rows,
        require_consensus=True,
    )

    assert records == []
    assert summary["skip_breakdown"]["event_type_mismatch"] == 1


def test_load_teacher_silver_samples_builds_training_ready_examples(tmp_path: Path):
    silver_path = tmp_path / "teacher_silver.jsonl"
    save_jsonl(
        [
            {
                "id": "train-1",
                "text": "甲公司完成10亿元融资，乙资本领投。",
                "event_list": [
                    {
                        "event_type": "企业融资",
                        "arguments": [
                            {"role": "融资方", "argument": "甲公司"},
                            {"role": "融资金额", "argument": "10亿元"},
                        ],
                    }
                ],
            }
        ],
        silver_path,
    )

    samples = load_teacher_silver_samples(
        silver_path,
        schema={"企业融资": ["融资方", "融资金额", "投资方"]},
        id_prefix="silver",
    )

    assert len(samples) == 1
    assert samples[0].id == "silver::train-1"
    assert samples[0].event_types == ["企业融资"]
    assert "抽取" in samples[0].prompt
    assert json.loads(samples[0].chosen)[0]["event_type"] == "企业融资"
