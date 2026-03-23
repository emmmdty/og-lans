from pathlib import Path


def test_main_and_evaluate_configure_shared_model_runtime():
    repo_root = Path(__file__).resolve().parents[1]
    main_text = (repo_root / "main.py").read_text(encoding="utf-8")
    evaluate_text = (repo_root / "evaluate.py").read_text(encoding="utf-8")

    assert "configure_model_download_runtime" in main_text
    assert "configure_model_download_runtime" in evaluate_text


def test_training_eval_and_scv_use_shared_model_resolver():
    repo_root = Path(__file__).resolve().parents[1]
    trainer_text = (repo_root / "src" / "oglans" / "trainer" / "unsloth_trainer.py").read_text(encoding="utf-8")
    evaluate_text = (repo_root / "evaluate.py").read_text(encoding="utf-8")
    scv_text = (repo_root / "src" / "oglans" / "utils" / "scv.py").read_text(encoding="utf-8")

    assert "resolve_model_name_or_path" in trainer_text
    assert "resolve_model_name_or_path" in evaluate_text
    assert "resolve_model_name_or_path" in scv_text
    assert "snapshot_download(model_name_or_path" not in trainer_text
    assert "snapshot_download(model_name_or_path" not in evaluate_text
