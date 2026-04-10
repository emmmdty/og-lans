#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OG-LANS: Ontology-Graph Loss-Aware Adaptive Negative Sampling

Training Entry Point for Event Extraction via Direct Preference Optimization

This module serves as the main training script for the OG-LANS framework,
which addresses the "Reasoning-Extraction Inconsistency" problem in LLM-based
Event Extraction through dynamic curriculum learning.

Academic Contribution:
    - OG-CNS: Ontology-Graph Driven Contrastive Negative Sampling
    - LANS: Loss-Aware Adaptive Negative Scheduling
    - CGA: Contrastive Gradient Amplification
    - SCV: Semantic Consistency Verification (NLI-based False Negative Filtering)

Usage:
    # Standard training
    python main.py --config configs/config.yaml --data_dir ./data/raw/DuEE-Fin

    # With experiment name
    python main.py --exp_name exp_v1 --data_dir ./data/raw/DuEE-Fin

    # CLI parameter override
    python main.py --training.max_steps 500 --algorithms.lans.enabled true

"""

import unsloth
import argparse
import sys
import os
import time
from oglans.utils import (
    setup_logger,
    build_contract_record,
    collect_runtime_manifest,
    compute_file_sha256,
    compute_json_sha256,
    build_run_manifest,
    save_json,
    configure_model_download_runtime,
    get_model_download_runtime_snapshot,
)
from oglans.utils.reproducibility import set_global_seed
from oglans.data import DuEEFinAdapter
from oglans.trainer import UnslothDPOTrainerWrapper, UnslothSFTTrainerWrapper
from oglans.config import ConfigManager
from oglans.utils.pathing import normalize_dataset_name, resolve_schema_path
try:
    from oglans.utils.research_protocol import resolve_stage_settings
except Exception:  # pragma: no cover - compatibility for tests stubbing utils package
    def resolve_stage_settings(*, comparison_cfg: dict | None = None, **_: object) -> dict:
        cfg = dict(comparison_cfg or {})
        return {
            "stage_mode": str(cfg.get("stage_mode", "single_pass")),
            "fewshot_selection_mode": str(cfg.get("fewshot_selection_mode", "dynamic")),
            "fewshot_pool_split": str(cfg.get("fewshot_pool_split", "train_fit")),
        }

try:
    from oglans.utils.training_protocol import select_training_fit_samples
except Exception:  # pragma: no cover - compatibility for tests stubbing utils package
    def select_training_fit_samples(
        samples,
        *,
        tune_ratio: float = 0.1,
        seed: int = 3407,
        split_manifest=None,
    ):
        del tune_ratio, seed, split_manifest
        return list(samples), {
            "seed": 3407,
            "tune_ratio": 0.1,
            "fit_ids": [],
            "tune_ids": [],
            "fit_count": len(samples),
            "tune_count": 0,
            "pool_split": "train_fit",
        }
try:
    from oglans.utils.teacher_silver import load_teacher_silver_samples
except Exception:  # pragma: no cover - compatibility for tests stubbing utils package
    def load_teacher_silver_samples(*args, **kwargs):
        raise RuntimeError("teacher_silver support is unavailable in the current test stub environment")
from oglans.data.prompt_builder import (
    PROMPT_BUILDER_VERSION,
    build_inference_prompt_payload,
)
try:
    from oglans.data.prompt_builder import resolve_prompt_settings
except ImportError:  # pragma: no cover - compatibility for tests stubbing prompt_builder
    def resolve_prompt_settings(
        *,
        default_prompt_variant: str = "zeroshot",
        default_num_examples: int = 3,
        **_: object,
    ) -> dict:
        prompt_variant = str(default_prompt_variant or "zeroshot")
        return {
            "prompt_variant": prompt_variant,
            "use_oneshot": prompt_variant == "fewshot",
            "fewshot_num_examples": int(default_num_examples) if prompt_variant == "fewshot" else 0,
        }
from oglans.utils.json_parser import NORMALIZATION_VERSION, PARSER_VERSION
from oglans.utils.model_profile import load_local_model_profile
import yaml
import torch

# Fix OOM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def create_trainer(config, samples):
    training_mode = str(config.get("training", {}).get("mode", "preference"))
    if training_mode == "sft":
        return UnslothSFTTrainerWrapper(config, samples)
    if training_mode == "preference":
        return UnslothDPOTrainerWrapper(config, samples)
    raise ValueError(f"Unsupported training.mode: {training_mode}")


def main():
    run_start_ts = time.time()
    cmdline = " ".join(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--data_dir", type=str, default=None, help="数据目录，不指定时使用默认 DuEE-Fin")
    parser.add_argument("--schema_path", type=str, default=None, help="可选：显式指定 schema 文件路径")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name (e.g., exp1)")
    args, unknown = parser.parse_known_args()

    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config, unknown)
    model_source = str(config.get("model", {}).get("source", "modelscope"))
    configure_model_download_runtime(
        os.path.dirname(os.path.abspath(__file__)),
        source=model_source,
    )

    # 全局随机种子设置 (Phase 3: Reproducibility)
    seed = config['project'].get('seed', 3407)
    deterministic = config['experiment'].get('deterministic', True)
    set_global_seed(seed, deterministic)

    # 数据目录：优先使用命令行参数，否则使用默认值
    data_dir = args.data_dir if args.data_dir else "./data/raw/DuEE-Fin"

    # 自动从 data_dir 提取数据集名字 (如 DuEE-Fin)
    dataset_name = normalize_dataset_name(data_dir)
    dataset_name_lower = dataset_name.lower().replace("-", "_")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_train_seed{seed}_p{os.getpid()}"
    config['project']['run_id'] = run_id

    project_cfg = config["project"]
    project_cfg.setdefault("output_dir", f"./logs/{dataset_name}/checkpoints")
    project_cfg.setdefault("logging_dir", f"./logs/{dataset_name}/tensorboard")
    project_cfg.setdefault("debug_data_dir", f"./logs/{dataset_name}/samples")
    project_cfg.setdefault("dataset_cache_dir", f"./data/processed/{dataset_name}")
    
    # Schema 路径：优先 CLI 显式指定，其次 data_dir 内推断，再回退到配置路径
    configured_schema_path = config['algorithms']['ds_cns'].get('taxonomy_path')
    resolved_schema_path, schema_candidates = resolve_schema_path(
        data_dir=data_dir,
        dataset_name=dataset_name,
        configured_schema_path=configured_schema_path,
        cli_schema_path=args.schema_path,
    )
    if not resolved_schema_path or not os.path.exists(resolved_schema_path):
        attempted = "\n".join([f"  - {os.path.abspath(p)}" for p in schema_candidates])
        raise FileNotFoundError(
            "无法定位 schema 文件。请通过 --schema_path 显式指定，或检查以下候选路径：\n"
            f"{attempted}"
        )
    config['algorithms']['ds_cns']['taxonomy_path'] = resolved_schema_path

    # 图缓存路径保持按数据集动态默认，避免跨数据集混用缓存
    config['algorithms']['ds_cns']['graph_cache_path'] = f"./data/schemas/{dataset_name_lower}_graph.gml"

    if args.exp_name:
        # 获取基础路径
        base_output = config['project']['output_dir']
        base_log = config['project']['logging_dir']
        base_debug = config['project']['debug_data_dir']
        base_cache = config['project']['dataset_cache_dir']
        
        # 将实验名拼接到路径后
        config['project']['output_dir'] = os.path.join(base_output, args.exp_name)
        config['project']['logging_dir'] = os.path.join(base_log, args.exp_name)
        config['project']['debug_data_dir'] = os.path.join(base_debug, args.exp_name)
        config['project']['dataset_cache_dir'] = os.path.join(base_cache, args.exp_name)
        
        print(f"🚀 Experiment Name: {args.exp_name}")
        print(f"📂 Output Dir: {config['project']['output_dir']}")

    # 落盘最终解析配置，便于复现实验
    os.makedirs(config['project']['output_dir'], exist_ok=True)
    os.makedirs(config['project']['logging_dir'], exist_ok=True)
    resolved_config_path = os.path.join(config['project']['output_dir'], "resolved_config.yaml")
    with open(resolved_config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    # 初始化日志
    logger = setup_logger("OGLANS", config['project']['logging_dir'])
    logger.info(f"Loaded config from {args.config}")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Data dir: {os.path.abspath(data_dir)}")
    logger.info(f"Schema path: {os.path.abspath(config['algorithms']['ds_cns']['taxonomy_path'])}")

    # 数据加载
    logger.info(">>> Stage 1: Data Loading")
    schema_path = config['algorithms']['ds_cns']['taxonomy_path']
    adapter = DuEEFinAdapter(data_dir, schema_path)
    samples = adapter.load_data("train")

    # [DEBUG] 如果配置了 max_samples，则截断数据
    max_samples = config['project'].get('max_samples')
    if max_samples and max_samples > 0:
        logger.info(f"🐛 Debug Mode: Limiting samples to {max_samples}")
        samples = samples[:max_samples]
    
    if not samples:
        logger.error("No samples loaded. Exiting.")
        sys.exit(1)

    # 【学术可复现性】设置确定性训练 (已通过 set_global_seed 处理)
    # if config.get('experiment', {}).get('deterministic', True):
    #    torch.backends.cudnn.deterministic = True
    #    torch.backends.cudnn.benchmark = False
    #    logger.info("🔒 确定性模式已启用 (cudnn.deterministic=True)")

    repo_dir = os.path.dirname(os.path.abspath(__file__))
    runtime_manifest = collect_runtime_manifest(
        repo_dir,
        package_names=["torch", "transformers", "trl", "unsloth", "datasets", "PyYAML"],
    )
    runtime_manifest["model_runtime"] = get_model_download_runtime_snapshot(source=model_source)
    config_hash_sha256 = compute_json_sha256(config)
    run_manifest_path = os.path.join(config['project']['output_dir'], "run_manifest.json")
    effective_model_path = str(config.get("model", {}).get("base_model"))
    contract = build_contract_record(
        model_profile=load_local_model_profile(config["model"]["profile"]).name,
        model_source=model_source,
        effective_model_path=effective_model_path,
    )

    manifest_status = "failed"
    error_message = None
    trainer = None
    training_mode = str(config.get("training", {}).get("mode", "preference"))
    comparison_cfg = config.get("comparison", {})

    def optional_file_sha256(path: str | None) -> str | None:
        if not path or not os.path.exists(path):
            return None
        return compute_file_sha256(path)

    configured_lans_enabled = bool(config.get("algorithms", {}).get("lans", {}).get("enabled", True))
    configured_scv_enabled = bool(config.get("algorithms", {}).get("scv", {}).get("enabled", False))
    stage_settings = resolve_stage_settings(comparison_cfg=comparison_cfg)
    train_tune_ratio = float(comparison_cfg.get("train_tune_ratio", 0.1))
    research_split_manifest_path = comparison_cfg.get("research_split_manifest_path")
    effective_train_samples, training_split_manifest = select_training_fit_samples(
        samples,
        tune_ratio=train_tune_ratio,
        seed=int(seed),
        split_manifest=research_split_manifest_path,
    )
    training_split_manifest_hash = (
        optional_file_sha256(research_split_manifest_path)
        if research_split_manifest_path
        else compute_json_sha256(training_split_manifest)
    )
    logger.info(
        "🧪 Training protocol: stage_mode=%s, pool_split=%s, train_fit=%s/%s",
        stage_settings["stage_mode"],
        stage_settings["fewshot_pool_split"],
        len(effective_train_samples),
        len(samples),
    )
    teacher_silver_cfg = config.get("training", {}).get("teacher_silver", {}) or {}
    teacher_silver_enabled = bool(teacher_silver_cfg.get("enabled", False))
    teacher_silver_path = teacher_silver_cfg.get("path")
    teacher_silver_max_samples = teacher_silver_cfg.get("max_samples")
    teacher_silver_id_prefix = str(teacher_silver_cfg.get("id_prefix", "teacher"))
    teacher_silver_samples = []
    if teacher_silver_enabled:
        if not teacher_silver_path:
            raise ValueError("training.teacher_silver.enabled=true requires training.teacher_silver.path")
        teacher_silver_samples = load_teacher_silver_samples(
            teacher_silver_path,
            schema=getattr(adapter, "schema", None),
            max_text_length=int(getattr(adapter, "max_text_length", 3500)),
            max_samples=teacher_silver_max_samples,
            id_prefix=teacher_silver_id_prefix,
        )
        logger.info(
            "🪙 Teacher silver loaded: count=%s, path=%s",
            len(teacher_silver_samples),
            os.path.abspath(str(teacher_silver_path)),
        )
    training_input_samples = list(effective_train_samples) + list(teacher_silver_samples)
    logger.info(
        "📦 Training input summary: gold_train_fit=%s, teacher_silver=%s, total=%s",
        len(effective_train_samples),
        len(teacher_silver_samples),
        len(training_input_samples),
    )

    protocol_path = comparison_cfg.get("eval_protocol_path")
    role_alias_path = comparison_cfg.get("role_alias_map_path")
    prompt_settings = resolve_prompt_settings(
        default_prompt_variant=str(comparison_cfg.get("prompt_variant", "zeroshot")),
        default_num_examples=int(comparison_cfg.get("fewshot_num_examples", 3)),
    )
    prompt_payload = None

    # 训练
    logger.info(">>> Stage 2: Training")
    try:
        trainer = create_trainer(config, training_input_samples)
        trainer.load_model()
        prompt_payload = build_inference_prompt_payload(
            text=samples[0].text if samples else "",
            tokenizer=getattr(trainer, "tokenizer", None),
            prompt_variant=prompt_settings["prompt_variant"],
            use_oneshot=prompt_settings["use_oneshot"],
            schema=getattr(trainer, "prompt_schema", None),
            num_examples=int(prompt_settings["fewshot_num_examples"]),
        )
        trainer.train(use_lans=configured_lans_enabled)
        manifest_status = "completed"
    except Exception as e:
        error_message = str(e)
        logger.exception(f"Training failed: {error_message}")
        raise
    finally:
        trainer_runtime_stats = (
            trainer.get_runtime_stats()
            if trainer and hasattr(trainer, "get_runtime_stats")
            else {}
        )
        run_manifest = build_run_manifest(
            task="train",
            status=manifest_status,
            meta={
                "run_id": run_id,
                "timestamp": timestamp,
                "dataset": dataset_name,
                "seed": seed,
                "deterministic": bool(deterministic),
                "command": cmdline,
                "config_path": os.path.abspath(args.config),
                "config_hash_sha256": config_hash_sha256,
                "exp_name": args.exp_name,
                "schema_path": os.path.abspath(config['algorithms']['ds_cns']['taxonomy_path']),
                "schema_candidates": [os.path.abspath(p) for p in schema_candidates],
                "overrides": unknown,
                "error": error_message,
                "training_mode": training_mode,
                "eval_protocol_path": os.path.abspath(protocol_path) if protocol_path else None,
                "eval_protocol_hash": optional_file_sha256(protocol_path),
                "role_alias_path": os.path.abspath(role_alias_path) if role_alias_path else None,
                "role_alias_hash": optional_file_sha256(role_alias_path),
                "prompt_variant": (
                    prompt_payload.get("prompt_variant")
                    if prompt_payload
                    else prompt_settings["prompt_variant"]
                ),
                "stage_mode": trainer_runtime_stats.get("training_protocol", {}).get(
                    "stage_mode",
                    stage_settings["stage_mode"],
                ),
                "fewshot_selection_mode": trainer_runtime_stats.get("training_protocol", {}).get(
                    "fewshot_selection_mode",
                    stage_settings["fewshot_selection_mode"],
                ),
                "fewshot_pool_split": trainer_runtime_stats.get("training_protocol", {}).get(
                    "fewshot_pool_split",
                    stage_settings["fewshot_pool_split"],
                ),
                "train_tune_ratio": train_tune_ratio,
                "research_split_manifest_path": (
                    os.path.abspath(research_split_manifest_path)
                    if research_split_manifest_path
                    else None
                ),
                "research_split_manifest_hash": training_split_manifest_hash,
                "configured_train_count": len(samples),
                "effective_gold_train_count": len(effective_train_samples),
                "teacher_silver_enabled": teacher_silver_enabled,
                "teacher_silver_path": (
                    os.path.abspath(str(teacher_silver_path))
                    if teacher_silver_path
                    else None
                ),
                "teacher_silver_hash": optional_file_sha256(str(teacher_silver_path)) if teacher_silver_path else None,
                "teacher_silver_count": len(teacher_silver_samples),
                "teacher_silver_max_samples": teacher_silver_max_samples,
                "total_training_input_count": len(training_input_samples),
                "effective_train_count": trainer_runtime_stats.get("training_protocol", {}).get(
                    "effective_train_count",
                    len(training_input_samples),
                ),
                "training_stage_breakdown": trainer_runtime_stats.get("training_protocol", {}).get(
                    "training_stage_breakdown",
                    {},
                ),
                "configured_lans_enabled": configured_lans_enabled,
                "effective_lans_enabled": trainer_runtime_stats.get("training_protocol", {}).get(
                    "effective_lans_enabled",
                    configured_lans_enabled and training_mode == "preference",
                ),
                "configured_scv_enabled": configured_scv_enabled,
                "effective_scv_enabled": trainer_runtime_stats.get("training_protocol", {}).get(
                    "effective_scv_enabled",
                    configured_scv_enabled and training_mode == "preference",
                ),
                "prompt_builder_version": str(comparison_cfg.get("prompt_builder_version", PROMPT_BUILDER_VERSION)),
                "parser_version": str(comparison_cfg.get("parser_version", PARSER_VERSION)),
                "normalization_version": str(comparison_cfg.get("normalization_version", NORMALIZATION_VERSION)),
            },
            artifacts={
                "output_dir": os.path.abspath(config['project']['output_dir']),
                "logging_dir": os.path.abspath(config['project']['logging_dir']),
                "debug_data_dir": os.path.abspath(config['project']['debug_data_dir']),
                "dataset_cache_dir": os.path.abspath(config['project']['dataset_cache_dir']),
                "resolved_config": os.path.abspath(resolved_config_path),
            },
            contract=contract,
            runtime={
                "wall_clock_seconds": round(time.time() - run_start_ts, 4),
                "phase_timings_seconds": trainer_runtime_stats.get("phase_timings_seconds", {}),
                "scv_runtime": trainer_runtime_stats.get("scv_runtime", {}),
                "lans_sampling": trainer_runtime_stats.get("lans_sampling", {}),
                "lans_runtime_mode": trainer_runtime_stats.get("lans_runtime_mode"),
            },
            runtime_manifest=runtime_manifest,
        )
        save_json(run_manifest_path, run_manifest)
        logger.info(f"🧾 Run manifest saved: {run_manifest_path}")

if __name__ == "__main__":
    main()
