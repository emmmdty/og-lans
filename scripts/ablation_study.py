#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OG-LANS 消融实验脚本 (Ablation Study for 2026 Publication)

本脚本用于系统性地执行消融实验，验证各组件的贡献度。
生成的结果可直接用于论文 Table 3 (消融分析)。

消融实验设置 (A1-A7):
- Full: OG-LANS 完整模型
- A1: w/o LANS (静态课程学习)
- A2: w/o SCV (无语义一致性验证)
- A3: Random Neg (随机负样本)
- A4: w/o EMA (无 EMA 能力平滑)
- A5: w/o CGA (无对比梯度放大)
- A6: w/o Ontology (无本体图语义距离)
- A7: Single-Granularity (仅事件级扰动)

Usage:
    python scripts/ablation_study.py --config configs/config.yaml --experiments A1,A2,A3

Author: OG-LANS Team
Date: 2026
"""

import os
import sys
import yaml
import copy
import json
import argparse
import subprocess
import statistics
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from oglans.config import ConfigManager
from oglans.data.prompt_builder import validate_prompt_variant
from oglans.utils.pathing import infer_dataset_name_from_config as infer_dataset_name_from_loaded_config

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class AblationConfig:
    """消融实验配置"""
    name: str
    tag: str
    description: str
    config_overrides: Dict = field(default_factory=dict)


# =====================================================
# 消融实验定义
# =====================================================
ABLATION_EXPERIMENTS = {
    "full": AblationConfig(
        name="Full Model",
        tag="full",
        description="OG-LANS 完整模型 (所有组件启用)",
        config_overrides={}
    ),
    
    "A1": AblationConfig(
        name="w/o LANS",
        tag="A1_no_lans",
        description="移除 LANS 动态调度，使用静态课程学习",
        config_overrides={
            "algorithms.lans.enabled": False
        }
    ),
    
    "A2": AblationConfig(
        name="w/o SCV",
        tag="A2_no_scv",
        description="移除语义一致性验证，不过滤假阴性",
        config_overrides={
            "algorithms.scv.enabled": False
        }
    ),
    
    "A3": AblationConfig(
        name="Random Neg",
        tag="A3_random_neg",
        description="随机负样本选择，不使用本体图距离",
        config_overrides={
            "algorithms.ds_cns.use_ontology_distance": False,
            "algorithms.lans.enabled": False
        }
    ),
    
    "A4": AblationConfig(
        name="w/o EMA",
        tag="A4_no_ema",
        description="移除 EMA 能力平滑，使用瞬时损失",
        config_overrides={
            "algorithms.lans.use_ema": False
        }
    ),
    
    "A5": AblationConfig(
        name="w/o CGA",
        tag="A5_no_cga",
        description="移除对比梯度放大机制",
        config_overrides={
            "algorithms.lans.use_cga": False
        }
    ),
    
    "A6": AblationConfig(
        name="w/o Ontology",
        tag="A6_no_ontology",
        description="移除本体图语义距离，仅使用字面相似度",
        config_overrides={
            "algorithms.ds_cns.use_ontology_distance": False
        }
    ),
    
    "A7": AblationConfig(
        name="Single-Granularity",
        tag="A7_single_granularity",
        description="仅使用事件级扰动，不使用多粒度策略",
        config_overrides={
            "algorithms.lans.granularity_weights.event_level": 1.0,
            "algorithms.lans.granularity_weights.argument_level": 0.0,
            "algorithms.lans.granularity_weights.value_level": 0.0
        }
    ),
}


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    return ConfigManager.load_config(config_path)


def parse_seeds(text: str) -> List[int]:
    seeds: List[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("No valid seeds parsed.")
    return seeds


def require_explicit_seeds(text: Optional[str]) -> List[int]:
    if text is None or not str(text).strip():
        raise ValueError("Ablation study requires explicit --seeds.")
    return parse_seeds(str(text))


def parse_prompt_modes(text: Optional[str]) -> List[str]:
    raw_modes = [token.strip() for token in str(text or "zeroshot").split(",") if token.strip()]
    if not raw_modes:
        raise ValueError("Ablation study requires at least one prompt mode.")
    modes: List[str] = []
    for mode in raw_modes:
        normalized = validate_prompt_variant(mode)
        if normalized not in modes:
            modes.append(normalized)
    return modes


def validate_eval_split(split: str) -> str:
    normalized = str(split or "").strip().lower()
    if normalized != "dev":
        raise ValueError("Ablation evaluation must use dev split.")
    return normalized


def build_seeded_experiment_name(tag: str, seed: int) -> str:
    return f"{tag}_s{int(seed)}"


def resolve_checkpoint_dir(
    *,
    project_root: Path,
    dataset_name: str,
    experiment_name: str,
) -> Path:
    return project_root / "logs" / dataset_name / "checkpoints" / experiment_name


def aggregate_seed_metrics(seed_results: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    successful = [
        result.get("metrics", {})
        for result in seed_results.values()
        if result.get("status") == "success" and isinstance(result.get("metrics"), dict)
    ]
    if not successful:
        return {}

    metric_names = sorted({key for row in successful for key in row.keys()})
    aggregated: Dict[str, Any] = {"n_success_runs": len(successful), "metrics": {}}
    for metric in metric_names:
        values = [float(row[metric]) for row in successful if metric in row]
        if not values:
            continue
        aggregated["metrics"][metric] = {
            "mean": round(statistics.fmean(values), 6),
            "std": round(statistics.pstdev(values), 6) if len(values) > 1 else 0.0,
            "n_runs": len(values),
        }
    return aggregated


def aggregate_seed_metrics_by_prompt_mode(
    seed_results: Dict[int, Dict[str, Any]],
    prompt_modes: Sequence[str],
) -> Dict[str, Any]:
    aggregated: Dict[str, Any] = {}
    for prompt_mode in prompt_modes:
        per_mode_results: Dict[int, Dict[str, Any]] = {}
        for seed, result in seed_results.items():
            if result.get("status") != "success":
                continue
            mode_result = result.get("eval_by_prompt_mode", {}).get(prompt_mode)
            if mode_result:
                per_mode_results[int(seed)] = {
                    "status": "success",
                    "metrics": mode_result.get("metrics", {}),
                }
        aggregated[prompt_mode] = aggregate_seed_metrics(per_mode_results)
    return aggregated


def save_config(config: Dict, output_path: str):
    """保存配置文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def apply_overrides(config: Dict, overrides: Dict) -> Dict:
    """应用配置覆盖"""
    config = copy.deepcopy(config)
    
    for key_path, value in overrides.items():
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    return config


def _resolve_path(path_like: str) -> Path:
    """将配置中的相对路径解析为项目根目录下的绝对路径。"""
    path_obj = Path(path_like)
    if path_obj.is_absolute():
        return path_obj
    return (PROJECT_ROOT / path_obj).resolve()


def _flatten_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    兼容新版/旧版 evaluate.py 指标结构，统一输出扁平字段。
    """
    cot_faith = metrics.get("cot_faithfulness", 0.0)
    if isinstance(cot_faith, dict):
        cot_faith = cot_faith.get("overall", 0.0)

    return {
        "doc_role_micro_f1": float(metrics.get("doc_role_micro_f1", metrics.get("academic_metrics", {}).get("doc_ee", {}).get("overall", {}).get("MicroF1", 0.0))),
        "doc_instance_micro_f1": float(metrics.get("doc_instance_micro_f1", metrics.get("academic_metrics", {}).get("doc_ee", {}).get("instance", {}).get("MicroF1", 0.0))),
        "doc_combination_micro_f1": float(metrics.get("doc_combination_micro_f1", metrics.get("academic_metrics", {}).get("doc_ee", {}).get("combination", {}).get("MicroF1", 0.0))),
        "doc_event_type_micro_f1": float(metrics.get("doc_event_type_micro_f1", metrics.get("academic_metrics", {}).get("doc_ee", {}).get("classification", {}).get("MicroF1", 0.0))),
        "strict_precision": float(metrics.get("strict_precision", metrics.get("strict", {}).get("precision", 0.0))),
        "strict_recall": float(metrics.get("strict_recall", metrics.get("strict", {}).get("recall", 0.0))),
        "strict_f1": float(metrics.get("strict_f1", metrics.get("strict", {}).get("f1", 0.0))),
        "hallucination_rate": float(metrics.get("hallucination_rate", metrics.get("hallucination", {}).get("sample_rate", 0.0))),
        "cot_faithfulness": float(cot_faith),
    }


def run_experiment(
    base_config_path: str,
    ablation: AblationConfig,
    output_dir: str,
    seed: int,
    eval_split: str = "dev",
    eval_batch_size: int = 16,
    eval_num_samples: Optional[int] = None,
    protocol_path: str = "configs/eval_protocol.yaml",
    role_alias_map: str = "configs/role_aliases_duee_fin.yaml",
    canonical_metric_mode: str = "analysis_only",
    report_primary_metric: str = "doc_role_micro_f1",
    prompt_modes: Optional[Sequence[str]] = None,
    fewshot_num_examples: int = 3,
    dry_run: bool = False
) -> Optional[Dict]:
    """
    运行单个消融实验
    
    Args:
        base_config_path: 基础配置文件路径
        ablation: 消融实验配置
        output_dir: 输出目录
        dry_run: 是否仅生成配置不执行
    
    Returns:
        实验结果字典
    """
    print(f"\n{'='*60}")
    print(f"Running Ablation: {ablation.name} ({ablation.tag})")
    print(f"Description: {ablation.description}")
    print(f"{'='*60}")
    
    eval_split = validate_eval_split(eval_split)

    # 加载并修改配置
    config = load_config(base_config_path)
    config = apply_overrides(config, ablation.config_overrides)
    config.setdefault("project", {})
    config["project"]["seed"] = int(seed)

    config.setdefault("experiment", {})
    config["experiment"]["ablation_tag"] = ablation.tag
    dataset_name = infer_dataset_name_from_loaded_config(config) or "DuEE-Fin"
    experiment_name = build_seeded_experiment_name(ablation.tag, seed)

    exp_output_dir = Path(output_dir) / experiment_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)

    # 保留原始配置文件名（例如 config_debug.yaml），避免 main.py 的 debug 判定失效
    exp_config_path = exp_output_dir / Path(base_config_path).name
    save_config(config, str(exp_config_path))
    print(f"Config saved to: {exp_config_path}")

    if dry_run:
        print("[DRY RUN] Skipping training/evaluation...")
        return {
            "status": "dry_run",
            "config_path": str(exp_config_path),
            "seed": int(seed),
            "experiment_name": experiment_name,
            "prompt_modes": list(prompt_modes or ["zeroshot"]),
        }

    train_log_path = exp_output_dir / "train.log"
    eval_log_path = exp_output_dir / "eval.log"

    train_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        "--config",
        str(exp_config_path),
        "--exp_name",
        experiment_name,
    ]
    print(f"Training command: {' '.join(train_cmd)}")
    print(f"Training log: {train_log_path}")

    try:
        with open(train_log_path, "w", encoding="utf-8") as train_log:
            train_log.write(f"$ {' '.join(train_cmd)}\n\n")
            train_result = subprocess.run(
                train_cmd,
                cwd=str(PROJECT_ROOT),
                stdout=train_log,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=3600 * 12,  # 12 小时
            )

        if train_result.returncode != 0:
            return {
                "status": "failed_train",
                "train_log": str(train_log_path),
                "returncode": train_result.returncode,
            }

        checkpoint_path = resolve_checkpoint_dir(
            project_root=PROJECT_ROOT,
            dataset_name=dataset_name,
            experiment_name=experiment_name,
        )
        if not checkpoint_path.exists():
            return {
                "status": "failed_checkpoint_missing",
                "train_log": str(train_log_path),
                "expected_checkpoint_dir": str(checkpoint_path),
            }

        prompt_modes = list(prompt_modes or ["zeroshot"])
        eval_results_by_mode: Dict[str, Dict[str, Any]] = {}
        for prompt_mode in prompt_modes:
            mode_suffix = prompt_mode if len(prompt_modes) > 1 else "default"
            eval_output_jsonl = exp_output_dir / (
                f"eval_results_{prompt_mode}.jsonl" if len(prompt_modes) > 1 else "eval_results.jsonl"
            )
            eval_log_path = exp_output_dir / (
                f"eval_{mode_suffix}.log" if len(prompt_modes) > 1 else "eval.log"
            )
            eval_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "evaluate.py"),
                "--config",
                str(exp_config_path),
                "--checkpoint",
                str(checkpoint_path),
                "--protocol",
                protocol_path,
                "--split",
                eval_split,
                "--batch_size",
                str(eval_batch_size),
                "--output_file",
                str(eval_output_jsonl),
                "--role_alias_map",
                role_alias_map,
                "--canonical_metric_mode",
                canonical_metric_mode,
                "--report_primary_metric",
                report_primary_metric,
                "--prompt_variant",
                prompt_mode,
            ]
            if prompt_mode == "fewshot":
                eval_cmd.extend(["--fewshot_num_examples", str(fewshot_num_examples)])
            if eval_num_samples is not None:
                eval_cmd.extend(["--num_samples", str(eval_num_samples)])

            print(f"Evaluation command [{prompt_mode}]: {' '.join(eval_cmd)}")
            print(f"Evaluation log [{prompt_mode}]: {eval_log_path}")

            with open(eval_log_path, "w", encoding="utf-8") as eval_log:
                eval_log.write(f"$ {' '.join(eval_cmd)}\n\n")
                eval_result = subprocess.run(
                    eval_cmd,
                    cwd=str(PROJECT_ROOT),
                    stdout=eval_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=3600 * 6,  # 6 小时
                )

            if eval_result.returncode != 0:
                return {
                    "status": "failed_eval",
                    "train_log": str(train_log_path),
                    "eval_log": str(eval_log_path),
                    "checkpoint": str(checkpoint_path),
                    "prompt_mode": prompt_mode,
                    "returncode": eval_result.returncode,
                }

            metrics_path = Path(str(eval_output_jsonl).replace(".jsonl", "_metrics.json"))
            summary_path = Path(str(eval_output_jsonl).replace(".jsonl", "_summary.json"))
            if not metrics_path.exists():
                return {
                    "status": "failed_metrics_missing",
                    "train_log": str(train_log_path),
                    "eval_log": str(eval_log_path),
                    "checkpoint": str(checkpoint_path),
                    "prompt_mode": prompt_mode,
                    "expected_metrics_file": str(metrics_path),
                }

            with open(metrics_path, "r", encoding="utf-8") as f:
                raw_metrics = json.load(f)
            flat_metrics = _flatten_metrics(raw_metrics)
            eval_results_by_mode[prompt_mode] = {
                "metrics": flat_metrics,
                "metrics_raw": raw_metrics,
                "eval_log": str(eval_log_path),
                "result_file": str(eval_output_jsonl),
                "metrics_file": str(metrics_path),
                "summary_file": str(summary_path),
                "prompt_variant": prompt_mode,
                "fewshot_num_examples": fewshot_num_examples if prompt_mode == "fewshot" else 0,
            }

        primary_mode = prompt_modes[0]
        primary_eval = eval_results_by_mode[primary_mode]

        return {
            "status": "success",
            "metrics": primary_eval["metrics"],
            "metrics_raw": primary_eval["metrics_raw"],
            "train_log": str(train_log_path),
            "eval_log": primary_eval["eval_log"],
            "checkpoint": str(checkpoint_path),
            "result_file": primary_eval["result_file"],
            "metrics_file": primary_eval["metrics_file"],
            "summary_file": primary_eval["summary_file"],
            "config_path": str(exp_config_path),
            "seed": int(seed),
            "experiment_name": experiment_name,
            "dataset_name": dataset_name,
            "prompt_modes": list(prompt_modes),
            "eval_by_prompt_mode": eval_results_by_mode,
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "train_log": str(train_log_path),
            "eval_log": str(eval_log_path),
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "train_log": str(train_log_path),
            "eval_log": str(eval_log_path),
        }


def generate_latex_table(results: Dict[str, Dict]) -> str:
    """
    生成 LaTeX 表格 (用于论文 Table 3)
    
    Args:
        results: 实验结果字典
    
    Returns:
        LaTeX 表格字符串
    """
    latex = []
    latex.append(r"\begin{table}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Ablation Study Results on DuEE-Fin}")
    latex.append(r"\label{tab:ablation}")
    latex.append(r"\begin{tabular}{l|ccc|cc}")
    latex.append(r"\toprule")
    latex.append(r"Model & Strict P & Strict R & Strict F1 & Hall. Rate $\downarrow$ & CoT Faith. \\")
    latex.append(r"\midrule")
    
    for exp_id, exp_config in ABLATION_EXPERIMENTS.items():
        row_source = results.get(exp_id, {})
        metrics_source: Optional[Dict[str, Any]] = None
        if row_source.get("status") == "success":
            metrics_source = row_source.get("metrics", {})
        elif row_source.get("aggregated", {}).get("metrics"):
            metrics_source = {
                metric: values.get("mean", 0.0)
                for metric, values in row_source["aggregated"]["metrics"].items()
            }

        if metrics_source:
            metrics = _flatten_metrics(metrics_source)

            p = metrics["strict_precision"] * 100
            r = metrics["strict_recall"] * 100
            f1 = metrics["strict_f1"] * 100
            hall = metrics["hallucination_rate"] * 100
            faith = metrics["cot_faithfulness"] * 100
            
            row = f"{exp_config.name} & {p:.1f} & {r:.1f} & {f1:.1f} & {hall:.1f} & {faith:.1f} \\\\"
            
            if exp_id == "full":
                row = r"\textbf{" + row.replace("\\\\", r"} \\")
            
            latex.append(row)
        else:
            latex.append(f"{exp_config.name} & - & - & - & - & - \\\\")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    return "\n".join(latex)


def main():
    parser = argparse.ArgumentParser(description="OG-LANS 消融实验")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                        help="基础配置文件路径")
    parser.add_argument("--experiments", type=str, default="all",
                        help="要运行的实验，逗号分隔 (e.g., A1,A2,A3) 或 'all'")
    parser.add_argument("--output_dir", type=str, default="./ablation_results",
                        help="结果输出目录")
    parser.add_argument("--seeds", type=str, default=None,
                        help="训练/评测 seeds，逗号分隔。必填。")
    parser.add_argument("--eval_split", type=str, default="dev",
                        help="评估数据划分 (默认: dev)")
    parser.add_argument("--eval_batch_size", type=int, default=16,
                        help="评估 batch size (默认: 16)")
    parser.add_argument("--eval_num_samples", type=int, default=None,
                        help="仅评估前 N 条样本 (默认: 全量)")
    parser.add_argument("--protocol", type=str, default="configs/eval_protocol.yaml",
                        help="评测 protocol 路径")
    parser.add_argument("--role_alias_map", type=str, default="configs/role_aliases_duee_fin.yaml",
                        help="角色别名表路径")
    parser.add_argument("--canonical_metric_mode", type=str, default="analysis_only",
                        choices=["off", "analysis_only", "apply_for_aux_metric"],
                        help="辅助规范化评测模式")
    parser.add_argument("--report_primary_metric", type=str, default="doc_role_micro_f1",
                        help="主汇报指标")
    parser.add_argument("--prompt_modes", type=str, default="zeroshot",
                        help="评测 prompt 模式，逗号分隔 (e.g., zeroshot,fewshot)")
    parser.add_argument("--fewshot_num_examples", type=int, default=3,
                        help="few-shot 示例数（默认: 3）")
    parser.add_argument("--dry_run", action="store_true",
                        help="仅生成配置，不执行训练")
    parser.add_argument("--generate_latex", action="store_true",
                        help="从已有结果生成 LaTeX 表格")
    
    args = parser.parse_args()
    
    # 确定要运行的实验
    if args.experiments.lower() == "all":
        experiments_to_run = list(ABLATION_EXPERIMENTS.keys())
    else:
        experiments_to_run = [e.strip() for e in args.experiments.split(",")]
    
    # 验证实验名称
    for exp in experiments_to_run:
        if exp not in ABLATION_EXPERIMENTS:
            print(f"[ERROR] Unknown experiment: {exp}")
            print(f"Available: {list(ABLATION_EXPERIMENTS.keys())}")
            sys.exit(1)
    
    print(f"{'='*60}")
    print(f"OG-LANS Ablation Study")
    print(f"Experiments to run: {experiments_to_run}")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*60}")

    validate_eval_split(args.eval_split)
    base_config = load_config(args.config)
    _ = base_config
    seeds = require_explicit_seeds(args.seeds)
    prompt_modes = parse_prompt_modes(args.prompt_modes)

    # 运行实验
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for exp_id in experiments_to_run:
        ablation = ABLATION_EXPERIMENTS[exp_id]
        per_seed_results: Dict[int, Dict[str, Any]] = {}
        for seed in seeds:
            result = run_experiment(
                args.config,
                ablation,
                args.output_dir,
                seed,
                args.eval_split,
                args.eval_batch_size,
                args.eval_num_samples,
                args.protocol,
                args.role_alias_map,
                args.canonical_metric_mode,
                args.report_primary_metric,
                prompt_modes,
                args.fewshot_num_examples,
                args.dry_run
            )
            per_seed_results[int(seed)] = result

        success_count = sum(1 for result in per_seed_results.values() if result.get("status") == "success")
        if len(seeds) == 1 and len(prompt_modes) == 1:
            summary_result = dict(per_seed_results[seeds[0]])
        else:
            summary_result = {
                "status": "success" if success_count == len(seeds) else "partial_failure",
                "aggregated": aggregate_seed_metrics(per_seed_results),
                "aggregated_by_prompt_mode": aggregate_seed_metrics_by_prompt_mode(
                    per_seed_results,
                    prompt_modes,
                ),
            }
        summary_result["per_seed"] = per_seed_results
        summary_result["seeds"] = [int(seed) for seed in seeds]
        summary_result["prompt_modes"] = list(prompt_modes)
        results[exp_id] = summary_result
    
    # 保存结果汇总
    summary_path = os.path.join(args.output_dir, f"ablation_summary_{timestamp}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "experiments": experiments_to_run,
            "seeds": [int(seed) for seed in seeds],
            "prompt_modes": list(prompt_modes),
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"Summary saved to: {summary_path}")
    
    # 生成 LaTeX 表格
    if args.generate_latex or not args.dry_run:
        latex_table = generate_latex_table(results)
        latex_path = os.path.join(args.output_dir, f"ablation_table_{timestamp}.tex")
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to: {latex_path}")
        print("\n" + latex_table)
    
    print(f"{'='*60}")
    print("Ablation study completed!")


if __name__ == "__main__":
    main()
