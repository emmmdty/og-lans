import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "oglans"
    / "utils"
    / "experiment_contract.py"
)
spec = importlib.util.spec_from_file_location("experiment_contract", str(MODULE_PATH))
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load experiment_contract from {MODULE_PATH}")
experiment_contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment_contract)


def _payload(**overrides):
    payload = {
        "model_family": "api",
        "model_kind": "api_model",
        "split": "dev",
        "primary_metric": "doc_role_micro_f1",
        "stage_mode": "two_stage_typed",
        "prompt_variant": "fewshot",
        "fewshot_num_examples": 3,
        "fewshot_selection_mode": "contrastive",
        "fewshot_pool_split": "train_fit",
        "train_tune_ratio": 0.1,
        "research_split_manifest_path": "/tmp/frozen.json",
        "research_split_manifest_hash": "a" * 64,
        "pipeline_mode": "e2e",
        "postprocess_profile": "event_probe_v2",
        "canonical_metric_mode": "analysis_only",
        "prompt_builder_version": "phase3_mvp_v2",
        "configured_prompt_builder_version": "phase3_mvp_v1",
        "parser_version": "phase3_mvp_v1",
        "configured_parser_version": "phase3_mvp_v1",
        "normalization_version": "phase3_mvp_v1",
        "configured_normalization_version": "phase3_mvp_v1",
        "protocol_hash": "b" * 64,
        "role_alias_hash": "c" * 64,
        "seed": 3407,
        "seed_effective": False,
        "token_usage_kind": "actual",
    }
    payload.update(overrides)
    return payload


def test_build_experiment_contract_requires_expected_fields_and_normalizes_stage_mode():
    contract = experiment_contract.build_experiment_contract(_payload())

    assert contract["stage_mode"] == "two_stage_per_type"
    assert contract["postprocess_profile"] == "event_probe_v2"
    assert contract["experiment_contract_hash"]


def test_build_experiment_contract_derived_compare_payload_uses_effective_semantic_versions():
    contract = experiment_contract.build_experiment_contract(_payload())

    compare = experiment_contract.build_compare_contract_payload(contract)

    assert compare["stage_mode"] == "two_stage_per_type"
    assert compare["postprocess_profile"] == "event_probe_v2"
    assert compare["prompt_builder_version"] == "phase3_mvp_v2"
    assert compare["parser_version"] == "phase3_mvp_v1"
    assert compare["normalization_version"] == "phase3_mvp_v1"
