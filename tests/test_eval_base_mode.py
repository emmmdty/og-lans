import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

if "dirtyjson" not in sys.modules:
    sys.modules["dirtyjson"] = SimpleNamespace(loads=json.loads)

EVAL_PATH = ROOT / "evaluate.py"
spec = importlib.util.spec_from_file_location("evaluate_module", str(EVAL_PATH))
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load evaluate module from {EVAL_PATH}")
evaluate_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_module)


def test_parse_args_supports_base_only_without_checkpoint():
    args = evaluate_module.parse_args(["--base_only"])
    assert args.base_only is True
    assert args.checkpoint is None


def test_parse_args_supports_stage_mode_and_fewshot_pool_split():
    args = evaluate_module.parse_args(
        [
            "--base_only",
            "--stage_mode",
            "two_stage",
            "--fewshot_selection_mode",
            "dynamic",
            "--fewshot_pool_split",
            "train_fit",
        ]
    )

    assert args.stage_mode == "two_stage"
    assert args.fewshot_selection_mode == "dynamic"
    assert args.fewshot_pool_split == "train_fit"


def test_parse_args_supports_research_split_manifest():
    args = evaluate_module.parse_args(
        [
            "--base_only",
            "--research_split_manifest",
            "configs/research_splits/frozen.json",
        ]
    )

    assert args.research_split_manifest == "configs/research_splits/frozen.json"


def test_parse_args_supports_summary_file_override():
    args = evaluate_module.parse_args(
        [
            "--base_only",
            "--summary_file",
            "logs/DuEE-Fin/eval_base/custom_summary.json",
        ]
    )

    assert args.summary_file == "logs/DuEE-Fin/eval_base/custom_summary.json"


def test_parse_args_supports_data_dir_and_schema_path():
    args = evaluate_module.parse_args(
        [
            "--base_only",
            "--data_dir",
            "/data/TJK/og-lans/data/raw/DuEE-Fin",
            "--schema_path",
            "/data/TJK/og-lans/data/raw/DuEE-Fin/duee_fin_event_schema.json",
        ]
    )

    assert args.data_dir == "/data/TJK/og-lans/data/raw/DuEE-Fin"
    assert args.schema_path == "/data/TJK/og-lans/data/raw/DuEE-Fin/duee_fin_event_schema.json"


def test_parse_args_with_unknown_preserves_cli_overrides():
    args, unknown = evaluate_module.parse_args_with_unknown(
        [
            "--checkpoint",
            "logs/DuEE-Fin/checkpoints/exp1",
            "--model.source",
            "local",
            "--model.base_model",
            "/tmp/local-model",
        ]
    )

    assert args.checkpoint == "logs/DuEE-Fin/checkpoints/exp1"
    assert unknown == [
        "--model.source",
        "local",
        "--model.base_model",
        "/tmp/local-model",
    ]


def test_infer_dataset_name_for_eval_prefers_config_when_no_checkpoint():
    cfg = {
        "algorithms": {
            "ds_cns": {
                "taxonomy_path": "./data/raw/DuEE-Fin/duee_fin_event_schema.json"
            }
        }
    }
    assert evaluate_module.infer_dataset_name_for_eval(cfg, checkpoint_path=None) == "DuEE-Fin"


def test_infer_dataset_name_for_eval_falls_back_when_checkpoint_tag_is_not_real_dataset(monkeypatch):
    cfg = {
        "algorithms": {
            "ds_cns": {
                "taxonomy_path": "./data/raw/DuEE-Fin/duee_fin_event_schema.json"
            }
        }
    }

    def fake_exists(path):
        return "DuEE-Fin" in str(path)

    monkeypatch.setattr(evaluate_module.os.path, "exists", fake_exists)

    assert (
        evaluate_module.infer_dataset_name_for_eval(
            cfg,
            checkpoint_path="/data/TJK/og-lans/logs/smoke/checkpoints/exp1",
        )
        == "DuEE-Fin"
    )


def test_resolve_eval_dataset_context_prefers_configured_schema_dir_over_checkpoint_tag():
    cfg = {
        "algorithms": {
            "ds_cns": {
                "taxonomy_path": "./data/raw/DuEE-Fin/duee_fin_event_schema.json"
            }
        }
    }

    ctx = evaluate_module.resolve_eval_dataset_context(
        cfg,
        checkpoint_path="/data/TJK/og-lans/logs/smoke/checkpoints/exp1",
    )

    assert ctx["dataset_name"] == "DuEE-Fin"
    assert ctx["data_path"].replace("\\", "/").endswith("data/raw/DuEE-Fin")
    assert ctx["schema_path"].replace("\\", "/").endswith(
        "data/raw/DuEE-Fin/duee_fin_event_schema.json"
    )


def test_resolve_eval_dataset_context_explicit_overrides_take_priority():
    cfg = {
        "algorithms": {
            "ds_cns": {
                "taxonomy_path": "./data/raw/DuEE-Fin/duee_fin_event_schema.json"
            }
        }
    }

    ctx = evaluate_module.resolve_eval_dataset_context(
        cfg,
        checkpoint_path="/data/TJK/og-lans/logs/smoke/checkpoints/exp1",
        data_dir_override="/tmp/custom_data/MyData",
        schema_path_override="/tmp/custom_data/MyData/mydata_event_schema.json",
    )

    assert ctx["dataset_name"] == "MyData"
    assert ctx["data_path"] == "/tmp/custom_data/MyData"
    assert ctx["schema_path"] == "/tmp/custom_data/MyData/mydata_event_schema.json"


def test_validate_eval_args_rejects_missing_checkpoint_when_not_base_only():
    args = SimpleNamespace(base_only=False, checkpoint=None)
    with pytest.raises(ValueError):
        evaluate_module.validate_eval_args(args)


def test_validate_eval_args_rejects_conflicting_base_only_and_checkpoint():
    args = SimpleNamespace(base_only=True, checkpoint="logs/DuEE-Fin/checkpoints/x")
    with pytest.raises(ValueError):
        evaluate_module.validate_eval_args(args)


def test_resolve_eval_model_path_uses_shared_resolver_for_cli_override(monkeypatch):
    captured = {}

    def fake_resolver(model_name_or_path, *, source, project_root):
        captured["model_name_or_path"] = model_name_or_path
        captured["source"] = source
        captured["project_root"] = project_root
        return "/tmp/resolved-model"

    monkeypatch.setattr(evaluate_module, "resolve_model_name_or_path", fake_resolver)

    resolved = evaluate_module.resolve_eval_model_path(
        "Qwen/Qwen3-4B-Instruct-2507",
        {"base_model": "unused", "source": "modelscope"},
        project_root="/repo",
    )

    assert resolved == "/tmp/resolved-model"
    assert captured == {
        "model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507",
        "source": "modelscope",
        "project_root": "/repo",
    }


def test_resolve_eval_model_path_falls_back_to_config_base_model(monkeypatch):
    captured = {}

    def fake_resolver(model_name_or_path, *, source, project_root):
        captured["model_name_or_path"] = model_name_or_path
        captured["source"] = source
        captured["project_root"] = project_root
        return "/tmp/resolved-config-model"

    monkeypatch.setattr(evaluate_module, "resolve_model_name_or_path", fake_resolver)

    resolved = evaluate_module.resolve_eval_model_path(
        None,
        {"base_model": "Qwen/Qwen3-4B-Instruct-2507", "source": "modelscope"},
        project_root="/repo",
    )

    assert resolved == "/tmp/resolved-config-model"
    assert captured == {
        "model_name_or_path": "Qwen/Qwen3-4B-Instruct-2507",
        "source": "modelscope",
        "project_root": "/repo",
    }


def test_main_forwards_cli_overrides_to_config_manager(monkeypatch):
    captured = {}

    def fake_load_config(self, path, overrides=None, validate_semantic=True):
        captured["path"] = path
        captured["overrides"] = overrides
        return {
            "evaluation": {"mode": "scored"},
            "model": {"source": "local", "profile": "qwen3_instruct"},
            "comparison": {},
            "inference": {},
            "algorithms": {
                "ds_cns": {
                    "taxonomy_path": "./data/raw/DuEE-Fin/duee_fin_event_schema.json"
                }
            },
        }

    class SentinelError(RuntimeError):
        pass

    def stop_after_config(_profile):
        raise SentinelError("stop after config load")

    fake_unsloth = ModuleType("unsloth")
    fake_unsloth.FastLanguageModel = object()

    monkeypatch.setattr(evaluate_module.ConfigManager, "load_config", fake_load_config)
    monkeypatch.setattr(evaluate_module, "load_local_model_profile", stop_after_config)
    monkeypatch.setitem(sys.modules, "unsloth", fake_unsloth)

    with pytest.raises(SentinelError, match="stop after config load"):
        evaluate_module.main(
            [
                "--checkpoint",
                "logs/DuEE-Fin/checkpoints/exp1",
                "--model.source",
                "local",
                "--model.base_model",
                "/tmp/local-model",
            ]
        )

    assert captured == {
        "path": "configs/config.yaml",
        "overrides": [
            "--model.source",
            "local",
            "--model.base_model",
            "/tmp/local-model",
        ],
    }
