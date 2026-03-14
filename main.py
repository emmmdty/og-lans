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
    collect_runtime_manifest,
    compute_json_sha256,
    build_run_manifest,
    save_json,
)
from oglans.utils.reproducibility import set_global_seed
from oglans.data import DuEEFinAdapter
from oglans.trainer import UnslothDPOTrainerWrapper
from oglans.config import ConfigManager
from oglans.utils.pathing import normalize_dataset_name, resolve_schema_path
import yaml
import torch

# Fix OOM fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


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
    
    # 【修复】检测是否为 debug 配置，保留配置文件中的路径设置
    is_debug_config = "debug" in args.config.lower()
    
    if is_debug_config:
        # Debug 模式：使用配置文件中的路径，仅动态替换 schema 路径
        print("🔧 检测到 Debug 配置，使用配置文件中的路径设置")
    else:
        # 正式训练：动态生成路径
        config['project']['output_dir'] = f"./logs/{dataset_name}/checkpoints"
        config['project']['logging_dir'] = f"./logs/{dataset_name}/tensorboard"
        config['project']['debug_data_dir'] = f"./logs/{dataset_name}/samples"
        config['project']['dataset_cache_dir'] = f"./data/processed/{dataset_name}"
    
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
    config_hash_sha256 = compute_json_sha256(config)
    run_manifest_path = os.path.join(config['project']['output_dir'], "run_manifest.json")

    manifest_status = "failed"
    error_message = None
    trainer = None

    # 训练
    logger.info(">>> Stage 2: Unsloth DPO Training")
    try:
        trainer = UnslothDPOTrainerWrapper(config, samples)
        trainer.load_model()
        trainer.train()
        manifest_status = "completed"
    except Exception as e:
        error_message = str(e)
        logger.exception(f"Training failed: {error_message}")
        raise
    finally:
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
            },
            artifacts={
                "output_dir": os.path.abspath(config['project']['output_dir']),
                "logging_dir": os.path.abspath(config['project']['logging_dir']),
                "debug_data_dir": os.path.abspath(config['project']['debug_data_dir']),
                "dataset_cache_dir": os.path.abspath(config['project']['dataset_cache_dir']),
                "resolved_config": os.path.abspath(resolved_config_path),
            },
            runtime={
                "wall_clock_seconds": round(time.time() - run_start_ts, 4),
                "phase_timings_seconds": (
                    trainer.get_runtime_stats().get("phase_timings_seconds")
                    if trainer and hasattr(trainer, "get_runtime_stats")
                    else {}
                ),
            },
            runtime_manifest=runtime_manifest,
        )
        save_json(run_manifest_path, run_manifest)
        logger.info(f"🧾 Run manifest saved: {run_manifest_path}")

if __name__ == "__main__":
    main()
