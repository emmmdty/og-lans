import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "oglans" / "utils" / "run_manifest.py"
spec = importlib.util.spec_from_file_location("run_manifest", str(MODULE_PATH))
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load run_manifest from {MODULE_PATH}")
run_manifest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_manifest)


build_run_manifest = run_manifest.build_run_manifest
build_contract_record = run_manifest.build_contract_record
compute_file_sha256 = run_manifest.compute_file_sha256
compute_json_sha256 = run_manifest.compute_json_sha256
append_validation_error = run_manifest.append_validation_error
collect_distributed_runtime_metadata = run_manifest.collect_distributed_runtime_metadata
filter_wrapper_cli_args = run_manifest.filter_wrapper_cli_args
load_effective_config_metadata = run_manifest.load_effective_config_metadata
resolve_semantic_version_meta = run_manifest.resolve_semantic_version_meta
make_validation_error = run_manifest.make_validation_error
save_json = run_manifest.save_json


def test_compute_json_sha256_is_stable_for_same_payload():
    payload = {"b": 2, "a": 1}
    h1 = compute_json_sha256(payload)
    h2 = compute_json_sha256({"a": 1, "b": 2})
    assert h1 == h2
    assert isinstance(h1, str)
    assert len(h1) == 64


def test_save_json_and_compute_file_sha256(tmp_path):
    target = tmp_path / "manifest.json"
    payload = {"task": "eval_checkpoint", "status": "completed"}
    save_json(target, payload)

    assert target.exists()
    loaded = json.loads(target.read_text(encoding="utf-8"))
    assert loaded == payload

    digest = compute_file_sha256(target)
    assert isinstance(digest, str)
    assert len(digest) == 64


def test_build_run_manifest_shape():
    payload = build_run_manifest(
        task="train",
        status="completed",
        meta={"run_id": "abc"},
        artifacts={"output_dir": "/tmp/out"},
        contract={"validation_status": "passed"},
        experiment_contract={"experiment_contract_hash": "h" * 64},
        runtime={"wall_clock_seconds": 1.23},
        runtime_manifest={"python": {"version": "3.10"}},
    )
    assert payload["task"] == "train"
    assert payload["status"] == "completed"
    assert payload["meta"]["run_id"] == "abc"
    assert payload["artifacts"]["output_dir"] == "/tmp/out"
    assert payload["contract"]["validation_status"] == "passed"
    assert payload["experiment_contract"]["experiment_contract_hash"] == "h" * 64
    assert payload["runtime"]["wall_clock_seconds"] == 1.23
    assert payload["runtime_manifest"]["python"]["version"] == "3.10"


def test_filter_wrapper_cli_args_drops_wrapper_only_options():
    args = [
        "--config",
        "configs/config.yaml",
        "--data_dir",
        "./data/raw/DuEE-Fin",
        "--schema_path",
        "./data/raw/DuEE-Fin/duee_fin_event_schema.json",
        "--exp_name",
        "A2_no_scv_s3047",
        "--project.seed",
        "3047",
        "--algorithms.scv.enabled",
        "false",
    ]

    filtered = filter_wrapper_cli_args(args)

    assert filtered == ["--project.seed", "3047", "--algorithms.scv.enabled", "false"]


def test_filter_wrapper_cli_args_drops_distributed_launcher_injections():
    args = [
        "--config",
        "configs/config.yaml",
        "--local-rank=0",
        "--local_rank=1",
        "--project.seed",
        "3047",
    ]

    filtered = filter_wrapper_cli_args(args)

    assert filtered == ["--project.seed", "3047"]


def test_collect_distributed_runtime_metadata_reads_real_topology_from_env():
    meta = collect_distributed_runtime_metadata(
        {
            "LOCAL_RANK": "1",
            "RANK": "3",
            "WORLD_SIZE": "8",
            "LOCAL_WORLD_SIZE": "4",
        }
    )

    assert meta["distributed"] is True
    assert meta["local_rank"] == 1
    assert meta["rank"] == 3
    assert meta["world_size"] == 8
    assert meta["local_world_size"] == 4


def test_resolve_semantic_version_meta_records_effective_and_configured_versions():
    meta = resolve_semantic_version_meta(
        comparison_cfg={
            "prompt_builder_version": "phase3_mvp_v1",
            "parser_version": "phase3_mvp_v1",
            "normalization_version": "phase3_mvp_v1",
        },
        effective_prompt_builder_version="phase3_mvp_v2",
        effective_parser_version="phase3_mvp_v1",
        effective_normalization_version="phase3_mvp_v1",
    )

    assert meta["prompt_builder_version"] == "phase3_mvp_v2"
    assert meta["configured_prompt_builder_version"] == "phase3_mvp_v1"
    assert meta["parser_version"] == "phase3_mvp_v1"
    assert meta["configured_parser_version"] == "phase3_mvp_v1"
    assert meta["normalization_version"] == "phase3_mvp_v1"
    assert meta["configured_normalization_version"] == "phase3_mvp_v1"


def test_load_effective_config_metadata_applies_cli_overrides(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
project:
  seed: 3407
model:
  profile: qwen3_instruct
  source: modelscope
training:
  mode: preference
  warm_start_from_checkpoint: ./logs/DuEE-Fin/checkpoints/bootstrap
  teacher_silver:
    enabled: true
    path: ./logs/DuEE-Fin/silver/train_fit_teacher.jsonl
    max_samples: 128
algorithms:
  lans:
    enabled: true
  scv:
    enabled: true
comparison:
  stage_mode: single_pass
  fewshot_selection_mode: dynamic
  fewshot_pool_split: train_fit
  train_tune_ratio: 0.1
  research_split_manifest_path: configs/research_splits/duee_fin_train_seed3407_tune0.1.json
  prompt_builder_version: phase3_mvp_v1
  parser_version: phase3_mvp_v1
  normalization_version: phase3_mvp_v1
evaluation:
  mode: scored
""".strip(),
        encoding="utf-8",
    )

    meta = load_effective_config_metadata(
        str(config_path),
        cli_args=[
            "--config",
            str(config_path),
            "--project.seed",
            "3047",
            "--training.mode",
            "sft",
            "--algorithms.scv.enabled",
            "false",
            "--comparison.stage_mode",
            "two_stage",
            "--comparison.fewshot_pool_split",
            "train",
        ],
    )

    assert meta["seed"] == 3047
    assert meta["training_mode"] == "sft"
    assert meta["scv_enabled"] is False
    assert meta["stage_mode"] == "two_stage"
    assert meta["fewshot_pool_split"] == "train"
    assert meta["fewshot_selection_mode"] == "dynamic"
    assert meta["train_tune_ratio"] == 0.1
    assert meta["research_split_manifest_path"].endswith("duee_fin_train_seed3407_tune0.1.json")
    assert meta["warm_start_from_checkpoint"].endswith("checkpoints/bootstrap")
    assert meta["resume_training_from"] is None
    assert meta["teacher_silver_enabled"] is True
    assert meta["teacher_silver_path"].endswith("train_fit_teacher.jsonl")
    assert meta["teacher_silver_max_samples"] == 128
    assert isinstance(meta["config_hash_sha256"], str)
    assert len(meta["config_hash_sha256"]) == 64


def test_build_contract_record_marks_validation_status():
    contract = build_contract_record(
        model_profile="qwen3_instruct",
        model_source="modelscope",
        effective_model_path="/tmp/model",
    )

    assert contract["contract_version"] == "strict_v1"
    assert contract["validation_status"] == "passed"
    assert contract["validation_errors"] == []


def test_append_validation_error_deduplicates_records():
    records = []

    append_validation_error(
        records,
        code="cpu_disable_4bit",
        message="CPU path disabled 4bit quantization",
        stage="eval",
    )
    append_validation_error(
        records,
        code="cpu_disable_4bit",
        message="CPU path disabled 4bit quantization",
        stage="eval",
    )
    append_validation_error(
        records,
        code="canonical_metrics_skipped_no_alias_map",
        message="Alias map missing; canonical metrics skipped",
        stage="eval",
    )

    assert len(records) == 2
    assert records[0]["code"] == "cpu_disable_4bit"
    assert records[1]["code"] == "canonical_metrics_skipped_no_alias_map"


def test_make_validation_error_captures_stage_and_details():
    record = make_validation_error(
        code="contract_violation",
        message="model profile mismatch",
        stage="train",
        details={"profile": "unknown"},
    )

    assert record["severity"] == "error"
    assert record["stage"] == "train"
    assert record["details"]["profile"] == "unknown"
