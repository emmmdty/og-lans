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
        assert 'MODELSCOPE_CACHE="${PROJECT_ROOT}/models"' in text, name
        assert '${MODELSCOPE_CACHE:-${PROJECT_ROOT}/models}' not in text, name
        assert "HF_HOME" not in text, name
        assert "HF_HUB_CACHE" not in text, name
        assert "HF_ASSETS_CACHE" not in text, name
        assert "HF_XET_CACHE" not in text, name


def test_run_eval_api_wrapper_sets_uv_cache_dir_for_uv_managed_runtime():
    text = _read_script("run_eval_api.sh")

    assert 'export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"' in text


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
        assert (
            "ConfigManager.load_config" in text
            or "load_effective_config_metadata" in text
            or "scripts/resolve_config_context.py" in text
        ), name


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


def test_run_eval_academic_disallows_weak_seed_and_continue_flags():
    text = _read_script("run_eval_academic.sh")
    assert "--allow-weak-seed-sweep" not in text
    assert "--continue-on-error" not in text


def test_python_script_entrypoints_do_not_parse_main_config_with_yaml_safe_load():
    ablation_text = _read_script("ablation_study.py")
    assert "yaml.safe_load" not in ablation_text
    assert "ConfigManager.load_config" in ablation_text


def test_run_train_wrapper_uses_effective_config_helper_for_manifest_provenance():
    text = _read_script("run_train.sh")
    assert "load_effective_config_metadata" in text
    assert 'launcher_manifest.json' in text
    assert 'OG_LANS_LAUNCHER_MANIFEST' in text


def test_run_train_wrapper_prefers_uv_managed_python():
    text = _read_script("run_train.sh")
    assert "uv run python" in text


def test_run_train_wrapper_prefers_active_conda_python_before_uv():
    text = _read_script("run_train.sh")
    assert 'if [[ -n "${CONDA_PREFIX:-}" ]]' in text
    assert 'echo "python"' in text
    assert text.index('if [[ -n "${CONDA_PREFIX:-}" ]]') < text.index('if command -v uv >/dev/null 2>&1; then')


def test_run_train_wrapper_falls_back_to_uv_when_no_conda_env_is_active():
    text = _read_script("run_train.sh")
    assert 'if command -v uv >/dev/null 2>&1; then' in text
    assert 'echo "uv run python"' in text


def test_run_train_wrapper_no_longer_hardcodes_gpu_banner():
    text = _read_script("run_train.sh")
    assert "Environment: A6000 (48GB) | CUDA 11.8 | torch 2.6.0" not in text
    assert "torch.cuda.get_device_name" in text or "nvidia-smi" in text


def test_ablation_script_no_longer_uses_heuristic_checkpoint_candidates():
    text = _read_script("ablation_study.py")
    assert "checkpoint_candidates" not in text
    assert "resolve_checkpoint_dir" in text


def test_eval_shell_wrappers_expose_prompt_variant_controls():
    base_text = _read_script("run_eval_base.sh")
    academic_text = _read_script("run_eval_academic.sh")

    assert "--prompt-variant" in base_text
    assert "--fewshot-num-examples" in base_text
    assert "--stage-mode" in base_text
    assert "--fewshot-selection-mode" in base_text
    assert "--fewshot-pool-split" in base_text
    assert "--train-tune-ratio" in base_text
    assert "--research-split-manifest" in base_text
    assert "--postprocess-profile" in base_text
    assert "--num-samples" in base_text
    assert "--num_samples" in base_text
    assert "--prompt-variant" in academic_text
    assert "--fewshot-num-examples" in academic_text
    assert "--postprocess-profile" in academic_text


def test_run_eval_api_wrapper_exposes_base_url_controls():
    text = _read_script("run_eval_api.sh")

    assert "--base-url" in text
    assert "--base_url" in text
    assert "--stage-mode" in text
    assert "--fewshot-selection-mode" in text
    assert "--fewshot-pool-split" in text
    assert "--train-tune-ratio" in text
    assert "--research-split-manifest" in text
    assert "--postprocess-profile" in text
    assert '--base_url "$BASE_URL"' in text


def test_run_eval_api_wrapper_does_not_force_hardcoded_model_override():
    text = _read_script("run_eval_api.sh")

    assert 'MODEL="deepseek-chat"' not in text
    assert 'if [[ -n "$MODEL" ]]; then' in text
    assert 'RUN_CMD+=(--model "$MODEL")' in text


def test_run_eval_api_wrapper_prefers_active_python_or_uv_managed_python():
    text = _read_script("run_eval_api.sh")

    assert 'if [[ -n "${CONDA_PREFIX:-}" ]]' in text
    assert 'if [[ -n "${VIRTUAL_ENV:-}" ]]' in text
    assert 'if command -v uv >/dev/null 2>&1; then' in text
    assert 'echo "uv run python"' in text


def test_run_eval_api_wrapper_uses_python_command_array_for_uv_runtime():
    text = _read_script("run_eval_api.sh")

    assert "resolve_python_cmd()" in text
    assert 'PYTHON_CMD=(uv run python)' in text
    assert '"${PYTHON_CMD[@]}"' in text


def test_run_eval_api_wrapper_executes_command_array_without_shell_reparse():
    text = _read_script("run_eval_api.sh")

    assert "RUN_CMD=(" in text
    assert 'bash -lc "$cmd"' not in text
    assert '"${RUN_CMD[@]}"' in text


def test_removed_auxiliary_shell_scripts_are_absent():
    removed = [
        ROOT / "scripts" / "run_debug.sh",
        ROOT / "scripts" / "run_parallel_eval_train.sh",
        ROOT / "scripts" / "run_train_tmux.sh",
        ROOT / "scripts" / "run_eval_local.sh",
    ]
    for path in removed:
        assert not path.exists(), str(path)
