from pathlib import Path


def test_main_passes_effective_lans_flag_to_trainer_train():
    repo_root = Path(__file__).resolve().parents[1]
    main_text = (repo_root / "main.py").read_text(encoding="utf-8")

    assert "trainer.train(use_lans=" in main_text


def test_main_records_training_stage_and_split_protocol_metadata():
    repo_root = Path(__file__).resolve().parents[1]
    main_text = (repo_root / "main.py").read_text(encoding="utf-8")

    assert '"stage_mode"' in main_text
    assert '"fewshot_pool_split"' in main_text
    assert '"effective_train_count"' in main_text
    assert '"effective_lans_enabled"' in main_text


def test_sft_training_uses_expanded_training_samples_and_disables_lans():
    repo_root = Path(__file__).resolve().parents[1]
    trainer_text = (repo_root / "src" / "oglans" / "trainer" / "unsloth_trainer.py").read_text(
        encoding="utf-8"
    )

    assert "expanded_samples = self._expand_training_samples()" in trainer_text
    assert '"effective_lans_enabled"] = False' in trainer_text
