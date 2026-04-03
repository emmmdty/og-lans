from pathlib import Path

import pytest

from oglans.data.adapter import DuEEFinAdapter
from oglans.utils.eval_protocol import resolve_primary_metric_value, validate_primary_metric


def test_adapter_load_data_missing_split_file_fails_fast(tmp_path):
    data_dir = tmp_path / "DuEE-Fin"
    data_dir.mkdir()
    schema_path = data_dir / "duee_fin_event_schema.json"
    schema_path.write_text(
        '{"event_type":"中标","role_list":[{"role":"中标公司"}]}\n',
        encoding="utf-8",
    )
    adapter = DuEEFinAdapter(data_path=str(data_dir), schema_path=str(schema_path))

    with pytest.raises(FileNotFoundError, match="duee_fin_dev.json"):
        adapter.load_data("dev")


def test_validate_primary_metric_rejects_unknown_metric():
    with pytest.raises(ValueError, match="Unsupported primary metric"):
        validate_primary_metric("parse_error_rate")


def test_resolve_primary_metric_value_requires_explicit_metric_presence():
    with pytest.raises(ValueError, match="Primary metric missing"):
        resolve_primary_metric_value({"strict_f1": 0.42}, "type_f1")


def test_evaluate_api_no_longer_silently_falls_back_to_strict_f1():
    repo_root = Path(__file__).resolve().parents[1]
    evaluate_api_text = (repo_root / "evaluate_api.py").read_text(encoding="utf-8")

    assert "回退到 strict_f1" not in evaluate_api_text
    assert "resolve_primary_metric_value" in evaluate_api_text


def test_trainer_invalid_preference_modes_fail_fast_instead_of_warning_fallback():
    repo_root = Path(__file__).resolve().parents[1]
    trainer_text = (repo_root / "src" / "oglans" / "trainer" / "unsloth_trainer.py").read_text(encoding="utf-8")

    assert "回退到 ipo" not in trainer_text
    assert "回退到 margin_bucket" not in trainer_text
    assert "Unsupported preference_mode" in trainer_text
    assert "Unsupported odpo_offset_source" in trainer_text


def test_checkpoint_eval_no_longer_records_runtime_fallbacks():
    repo_root = Path(__file__).resolve().parents[1]
    evaluate_text = (repo_root / "evaluate.py").read_text(encoding="utf-8")

    assert "append_fallback_record" not in evaluate_text
    assert "derive_degraded_modes" not in evaluate_text
    assert "自动禁用 load_in_4bit" not in evaluate_text
    assert "canonical 指标" in evaluate_text


def test_training_no_longer_overrides_online_lans_workers_or_runtime_mode():
    repo_root = Path(__file__).resolve().parents[1]
    trainer_text = (repo_root / "src" / "oglans" / "trainer" / "unsloth_trainer.py").read_text(encoding="utf-8")

    assert "falling back to online_iterable" not in trainer_text
    assert "dataloader_num_workers_forced_zero" not in trainer_text
    assert "requires dataloader_num_workers=0" in trainer_text
