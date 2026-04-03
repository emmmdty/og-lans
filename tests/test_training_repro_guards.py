from pathlib import Path


def test_trainer_no_longer_hardcodes_peft_random_state():
    repo_root = Path(__file__).resolve().parents[1]
    trainer_text = (repo_root / "src" / "oglans" / "trainer" / "unsloth_trainer.py").read_text(encoding="utf-8")

    assert "random_state = 3407" not in trainer_text
    assert "random_state = int(self.config['project']['seed'])" in trainer_text


def test_training_prompt_payload_paths_pass_schema():
    repo_root = Path(__file__).resolve().parents[1]
    main_text = (repo_root / "main.py").read_text(encoding="utf-8")
    trainer_text = (repo_root / "src" / "oglans" / "trainer" / "unsloth_trainer.py").read_text(encoding="utf-8")

    assert "schema=getattr(trainer, \"prompt_schema\", None)" in main_text
    assert "schema=self.prompt_schema" in trainer_text


def test_main_no_longer_uses_debug_filename_heuristic():
    repo_root = Path(__file__).resolve().parents[1]
    main_text = (repo_root / "main.py").read_text(encoding="utf-8")

    assert '"debug" in args.config.lower()' not in main_text
    assert "project_cfg.setdefault" in main_text


def test_trainer_local_model_load_enforces_local_files_only():
    repo_root = Path(__file__).resolve().parents[1]
    trainer_text = (repo_root / "src" / "oglans" / "trainer" / "unsloth_trainer.py").read_text(encoding="utf-8")

    assert "build_unsloth_from_pretrained_kwargs" in trainer_text
