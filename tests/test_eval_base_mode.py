import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_infer_dataset_name_for_eval_prefers_config_when_no_checkpoint():
    cfg = {
        "algorithms": {
            "ds_cns": {
                "taxonomy_path": "./data/raw/DuEE-Fin/duee_fin_event_schema.json"
            }
        }
    }
    assert evaluate_module.infer_dataset_name_for_eval(cfg, checkpoint_path=None) == "DuEE-Fin"


def test_validate_eval_args_rejects_missing_checkpoint_when_not_base_only():
    args = SimpleNamespace(base_only=False, checkpoint=None)
    with pytest.raises(ValueError):
        evaluate_module.validate_eval_args(args)


def test_validate_eval_args_rejects_conflicting_base_only_and_checkpoint():
    args = SimpleNamespace(base_only=True, checkpoint="logs/DuEE-Fin/checkpoints/x")
    with pytest.raises(ValueError):
        evaluate_module.validate_eval_args(args)
