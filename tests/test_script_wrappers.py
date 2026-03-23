from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read_script(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_shell_wrappers_export_modelscope_runtime_defaults():
    targets = [
        "run_train.sh",
        "run_eval_base.sh",
        "run_eval_api.sh",
        "run_eval_academic.sh",
    ]
    for name in targets:
        text = _read_script(name)
        assert "MODELSCOPE_CACHE" in text, name


def test_shell_wrappers_do_not_parse_config_with_yaml_safe_load():
    targets = [
        "run_train.sh",
        "run_eval_base.sh",
        "run_eval_api.sh",
        "run_eval_academic.sh",
    ]
    for name in targets:
        text = _read_script(name)
        assert "yaml.safe_load" not in text, name
        assert "ConfigManager.load_config" in text or "scripts/resolve_config_context.py" in text, name


def test_run_train_has_standard_help_surface():
    text = _read_script("run_train.sh")
    assert "usage()" in text
    assert "--help" in text


def test_other_shell_entrypoints_have_help_surface():
    targets = [
        "run_eval_base.sh",
        "run_eval_api.sh",
        "run_eval_academic.sh",
    ]
    for name in targets:
        text = _read_script(name)
        assert "usage()" in text or "Usage:" in text, name
        assert "--help" in text, name


def test_python_script_entrypoints_do_not_parse_main_config_with_yaml_safe_load():
    ablation_text = _read_script("ablation_study.py")
    assert "yaml.safe_load" not in ablation_text
    assert "ConfigManager.load_config" in ablation_text


def test_removed_auxiliary_shell_scripts_are_absent():
    removed = [
        ROOT / "scripts" / "run_debug.sh",
        ROOT / "scripts" / "run_parallel_eval_train.sh",
        ROOT / "scripts" / "run_train_tmux.sh",
        ROOT / "scripts" / "run_eval_local.sh",
    ]
    for path in removed:
        assert not path.exists(), str(path)
