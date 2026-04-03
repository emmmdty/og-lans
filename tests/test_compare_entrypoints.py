from pathlib import Path


def test_evaluation_entrypoints_delegate_to_shared_compare_layers():
    repo_root = Path(__file__).resolve().parents[1]
    evaluate_text = (repo_root / "evaluate.py").read_text(encoding="utf-8")
    evaluate_api_text = (repo_root / "evaluate_api.py").read_text(encoding="utf-8")

    assert "shared_load_eval_protocol" in evaluate_text
    assert "shared_load_role_alias_map" in evaluate_text
    assert "parse_event_list_strict_with_diagnostics" in evaluate_text
    assert "build_inference_prompt_payload" in evaluate_text
    assert "ConfigManager" in evaluate_text
    assert "DEFAULT_EVAL_PROTOCOL" not in evaluate_text
    assert "yaml.safe_load(f)" not in evaluate_text

    assert "shared_load_eval_protocol" in evaluate_api_text
    assert "shared_load_role_alias_map" in evaluate_api_text
    assert "normalize_parsed_events" in evaluate_api_text
    assert "build_inference_prompt_payload" in evaluate_api_text
    assert "ConfigManager" in evaluate_api_text
    assert "DEFAULT_EVAL_PROTOCOL" not in evaluate_api_text
    assert "yaml.safe_load(f)" not in evaluate_api_text


def test_checkpoint_evaluation_no_longer_uses_eval_local_task_name():
    repo_root = Path(__file__).resolve().parents[1]
    evaluate_text = (repo_root / "evaluate.py").read_text(encoding="utf-8")

    assert '"eval_local"' not in evaluate_text
    assert '"eval_checkpoint"' in evaluate_text
    assert '"qwen_base_local"' not in evaluate_text


def test_checkpoint_evaluation_local_source_enforces_local_files_only():
    repo_root = Path(__file__).resolve().parents[1]
    evaluate_text = (repo_root / "evaluate.py").read_text(encoding="utf-8")

    assert "build_unsloth_from_pretrained_kwargs" in evaluate_text
