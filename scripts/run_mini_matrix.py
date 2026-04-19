#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run a mini end-to-end comparison matrix for base/full/ablations on a small sample subset.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from oglans.config import ConfigManager
from oglans.utils.experiment_contract import extract_experiment_contract
from oglans.utils.pathing import infer_dataset_name_from_config as infer_dataset_name_from_loaded_config


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACADEMIC_EVAL_PATH = PROJECT_ROOT / "src" / "oglans" / "utils" / "academic_eval.py"
ABALATION_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "ablation_study.py"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


academic_eval = _load_module("academic_eval", ACADEMIC_EVAL_PATH)
aggregate_runs = academic_eval.aggregate_runs
build_significance_metadata = academic_eval.build_significance_metadata
extract_report_metrics = academic_eval.extract_report_metrics
MIN_SIGNIFICANCE_PAIRS = academic_eval.MIN_SIGNIFICANCE_PAIRS
paired_permutation_pvalue = academic_eval.paired_permutation_pvalue

ablation_mod = _load_module("ablation_study", ABALATION_SCRIPT_PATH)
ABLATION_EXPERIMENTS = ablation_mod.ABLATION_EXPERIMENTS
apply_overrides = ablation_mod.apply_overrides
build_seeded_experiment_name = ablation_mod.build_seeded_experiment_name
parse_prompt_modes = ablation_mod.parse_prompt_modes
parse_seeds = ablation_mod.parse_seeds
resolve_checkpoint_dir = ablation_mod.resolve_checkpoint_dir
save_config = ablation_mod.save_config
validate_eval_split = ablation_mod.validate_eval_split
TARGET_METRICS = (
    "doc_role_micro_f1",
    "doc_instance_micro_f1",
    "doc_combination_micro_f1",
    "doc_event_type_micro_f1",
    "strict_f1",
    "relaxed_f1",
    "type_f1",
    "schema_compliance_rate",
    "hallucination_rate",
)


def parse_experiments(text: str) -> List[str]:
    if str(text).strip().lower() == "all":
        return list(ABLATION_EXPERIMENTS.keys())
    experiments = [token.strip() for token in str(text).split(",") if token.strip()]
    if not experiments:
        raise ValueError("No valid experiments parsed.")
    for exp in experiments:
        if exp not in ABLATION_EXPERIMENTS:
            raise ValueError(f"Unknown experiment: {exp}")
    return experiments


def infer_dataset_name_from_config(config_path: str) -> str:
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()
    cfg = ConfigManager.load_config(str(cfg_path))
    return infer_dataset_name_from_loaded_config(cfg) or "DuEE-Fin"


def build_training_overrides(
    *,
    seed: int,
    train_num_samples: int,
    model_name_or_path: Optional[str],
    model_source: Optional[str],
    scv_model_name_or_path: Optional[str],
    scv_source: Optional[str],
    train_batch_size: Optional[int],
    gradient_accumulation_steps: Optional[int],
    num_train_epochs: Optional[int],
    max_steps: Optional[int],
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {
        "project.seed": int(seed),
        "project.max_samples": int(train_num_samples),
    }
    if model_name_or_path:
        overrides["model.base_model"] = model_name_or_path
    if model_source:
        overrides["model.source"] = model_source
    if scv_model_name_or_path:
        overrides["algorithms.scv.nli_model"] = scv_model_name_or_path
    if scv_source:
        overrides["algorithms.scv.source"] = scv_source
    if train_batch_size is not None:
        overrides["training.per_device_train_batch_size"] = int(train_batch_size)
    if gradient_accumulation_steps is not None:
        overrides["training.gradient_accumulation_steps"] = int(gradient_accumulation_steps)
    if num_train_epochs is not None:
        overrides["training.num_train_epochs"] = int(num_train_epochs)
    if max_steps is not None:
        overrides["training.max_steps"] = int(max_steps)
    return overrides


def build_eval_command(
    *,
    config_path: Path,
    protocol_path: str,
    role_alias_map: str,
    output_file: Path,
    prompt_variant: str,
    fewshot_num_examples: int,
    eval_num_samples: Optional[int],
    eval_batch_size: int,
    seed: int,
    report_primary_metric: str,
    canonical_metric_mode: str,
    model_name_or_path: Optional[str],
    model_source: Optional[str],
    checkpoint_path: Optional[str],
    split: str = "dev",
) -> List[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "evaluate.py"),
        "--config",
        str(config_path),
        "--protocol",
        protocol_path,
        "--role_alias_map",
        role_alias_map,
        "--seed",
        str(seed),
        "--split",
        split,
        "--batch_size",
        str(eval_batch_size),
        "--output_file",
        str(output_file),
        "--canonical_metric_mode",
        canonical_metric_mode,
        "--report_primary_metric",
        report_primary_metric,
        "--prompt_variant",
        prompt_variant,
    ]
    if model_name_or_path:
        cmd.extend(["--model_name_or_path", model_name_or_path])
    if model_source:
        cmd.extend(["--model.source", model_source])
    if eval_num_samples is not None:
        cmd.extend(["--num_samples", str(eval_num_samples)])
    if prompt_variant == "fewshot":
        cmd.extend(["--fewshot_num_examples", str(fewshot_num_examples)])
    if checkpoint_path:
        cmd.extend(["--checkpoint", checkpoint_path])
    else:
        cmd.append("--base_only")
    return cmd


def run_logged_command(
    cmd: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout_seconds: int,
) -> Tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(str(part) for part in cmd)}\n\n")
        result = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_seconds,
        )
    return int(result.returncode), time.time() - start


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_metric_row(summary_or_metrics: Dict[str, Any], seed: int) -> Dict[str, float]:
    flat = extract_report_metrics(summary_or_metrics, required_metrics=TARGET_METRICS)
    row = {metric: float(flat[metric]) for metric in TARGET_METRICS}
    row["seed"] = float(seed)
    return row


def aggregate_mode_results(per_seed: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    rows = [build_metric_row(summary, seed) for seed, summary in sorted(per_seed.items())]
    metric_names = [name for name in TARGET_METRICS if rows and all(name in row for row in rows)]
    return aggregate_runs(rows, metric_names)


def compute_significance(
    *,
    by_run_prompt_seed: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]],
    prompt_modes: Sequence[str],
    seeds: Sequence[int],
    primary_metric: str,
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Any]]:
    significance: Dict[str, Dict[str, Dict[str, Any]]] = {}
    pair_counts: List[int] = []
    if "base" not in by_run_prompt_seed or "full" not in by_run_prompt_seed:
        return significance, build_significance_metadata(pair_counts)

    for prompt_mode in prompt_modes:
        base_seed_map = by_run_prompt_seed.get("base", {}).get(prompt_mode, {})
        full_seed_map = by_run_prompt_seed.get("full", {}).get(prompt_mode, {})
        common_seeds = sorted(set(base_seed_map) & set(full_seed_map))
        if common_seeds != sorted(int(seed) for seed in seeds):
            continue
        pair_counts.append(len(common_seeds))
        if len(common_seeds) < MIN_SIGNIFICANCE_PAIRS:
            continue
        metrics = [primary_metric] + [m for m in ("strict_f1", "type_f1") if m != primary_metric]
        pair_key = f"base_vs_full/{prompt_mode}"
        significance[pair_key] = {}
        for metric in metrics:
            baseline_scores = [float(build_metric_row(base_seed_map[seed], seed)[metric]) for seed in common_seeds]
            improved_scores = [float(build_metric_row(full_seed_map[seed], seed)[metric]) for seed in common_seeds]
            significance[pair_key][metric] = paired_permutation_pvalue(
                baseline_scores=baseline_scores,
                improved_scores=improved_scores,
                seed=3407,
            )
    return significance, build_significance_metadata(pair_counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a mini comparison matrix on a small sample subset.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--protocol", type=str, default="configs/eval_protocol.yaml")
    parser.add_argument("--role_alias_map", type=str, default="configs/role_aliases_duee_fin.yaml")
    parser.add_argument("--experiments", type=str, default="all")
    parser.add_argument("--seeds", type=str, required=True)
    parser.add_argument("--prompt_modes", type=str, default="zeroshot,fewshot")
    parser.add_argument("--fewshot_num_examples", type=int, default=3)
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test", "train"])
    parser.add_argument("--train_num_samples", type=int, default=12)
    parser.add_argument("--eval_num_samples", type=int, default=12)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--model_source", type=str, default="local")
    parser.add_argument("--scv_model", type=str, default=None)
    parser.add_argument("--scv_source", type=str, default="local")
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--num_train_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--canonical_metric_mode", type=str, default="analysis_only")
    parser.add_argument("--report_primary_metric", type=str, default="doc_role_micro_f1")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    validate_eval_split(args.split)
    seeds = parse_seeds(args.seeds)
    prompt_modes = parse_prompt_modes(args.prompt_modes)
    experiments = parse_experiments(args.experiments)
    dataset_name = infer_dataset_name_from_config(args.config)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suite_dir = (
        Path(args.output_dir)
        if args.output_dir
        else PROJECT_ROOT / "logs" / dataset_name / "eval_academic" / f"mini_matrix_{timestamp}"
    )
    suite_dir.mkdir(parents=True, exist_ok=True)
    config_dir = suite_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    base_config = ConfigManager.load_config(args.config)

    records: List[Dict[str, Any]] = []
    by_run_prompt_seed: Dict[str, Dict[str, Dict[int, Dict[str, Any]]]] = {}

    # Base-only reference
    for prompt_mode in prompt_modes:
        for seed in seeds:
            run_dir = suite_dir / "base" / prompt_mode / f"seed{seed}"
            output_file = run_dir / "eval_results.jsonl"
            log_file = run_dir / "run.log"
            cmd = build_eval_command(
                config_path=Path(args.config),
                protocol_path=args.protocol,
                role_alias_map=args.role_alias_map,
                output_file=output_file,
                prompt_variant=prompt_mode,
                fewshot_num_examples=args.fewshot_num_examples,
                eval_num_samples=args.eval_num_samples,
                eval_batch_size=args.eval_batch_size,
                seed=seed,
                report_primary_metric=args.report_primary_metric,
                canonical_metric_mode=args.canonical_metric_mode,
                model_name_or_path=args.base_model,
                model_source=args.model_source,
                checkpoint_path=None,
                split=args.split,
            )
            returncode, duration = run_logged_command(
                cmd,
                cwd=PROJECT_ROOT,
                log_path=log_file,
                timeout_seconds=3600,
            )
            summary_path = Path(str(output_file).replace(".jsonl", "_summary.json"))
            record = {
                "run_key": "base",
                "prompt_variant": prompt_mode,
                "seed": seed,
                "returncode": returncode,
                "duration_seconds": duration,
                "log_file": str(log_file),
                "summary_file": str(summary_path),
                "command": cmd,
            }
            records.append(record)
            if returncode != 0:
                raise RuntimeError(f"Base evaluation failed: prompt={prompt_mode} seed={seed} log={log_file}")
            by_run_prompt_seed.setdefault("base", {}).setdefault(prompt_mode, {})[seed] = load_json(summary_path)

    for experiment in experiments:
        ablation = ABLATION_EXPERIMENTS[experiment]
        for seed in seeds:
            training_overrides = build_training_overrides(
                seed=seed,
                train_num_samples=args.train_num_samples,
                model_name_or_path=args.base_model,
                model_source=args.model_source,
                scv_model_name_or_path=args.scv_model,
                scv_source=args.scv_source,
                train_batch_size=args.train_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                num_train_epochs=args.num_train_epochs,
                max_steps=args.max_steps,
            )
            config_payload = apply_overrides(base_config, ablation.config_overrides)
            config_payload = apply_overrides(config_payload, training_overrides)
            config_payload.setdefault("comparison", {})
            config_payload["comparison"]["prompt_variant"] = "zeroshot"
            config_payload["comparison"]["fewshot_num_examples"] = int(args.fewshot_num_examples)

            experiment_name = build_seeded_experiment_name(f"mini_{ablation.tag}", seed)
            config_path = config_dir / f"{experiment_name}.yaml"
            save_config(config_payload, str(config_path))

            train_run_dir = suite_dir / experiment / f"seed{seed}"
            train_log = train_run_dir / "train.log"
            train_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "main.py"),
                "--config",
                str(config_path),
                "--exp_name",
                experiment_name,
            ]
            returncode, duration = run_logged_command(
                train_cmd,
                cwd=PROJECT_ROOT,
                log_path=train_log,
                timeout_seconds=3600 * 6,
            )
            train_record = {
                "run_key": experiment,
                "seed": seed,
                "stage": "train",
                "returncode": returncode,
                "duration_seconds": duration,
                "log_file": str(train_log),
                "command": train_cmd,
            }
            records.append(train_record)
            if returncode != 0:
                raise RuntimeError(f"Training failed: experiment={experiment} seed={seed} log={train_log}")

            checkpoint_path = resolve_checkpoint_dir(
                project_root=PROJECT_ROOT,
                dataset_name=dataset_name,
                experiment_name=experiment_name,
            )
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Missing checkpoint directory: {checkpoint_path}")

            for prompt_mode in prompt_modes:
                eval_run_dir = train_run_dir / prompt_mode
                output_file = eval_run_dir / "eval_results.jsonl"
                eval_log = eval_run_dir / "eval.log"
                eval_cmd = build_eval_command(
                    config_path=config_path,
                    protocol_path=args.protocol,
                    role_alias_map=args.role_alias_map,
                    output_file=output_file,
                    prompt_variant=prompt_mode,
                    fewshot_num_examples=args.fewshot_num_examples,
                    eval_num_samples=args.eval_num_samples,
                    eval_batch_size=args.eval_batch_size,
                    seed=seed,
                    report_primary_metric=args.report_primary_metric,
                    canonical_metric_mode=args.canonical_metric_mode,
                    model_name_or_path=args.base_model,
                    model_source=args.model_source,
                    checkpoint_path=str(checkpoint_path),
                    split=args.split,
                )
                returncode, duration = run_logged_command(
                    eval_cmd,
                    cwd=PROJECT_ROOT,
                    log_path=eval_log,
                    timeout_seconds=3600 * 6,
                )
                summary_path = Path(str(output_file).replace(".jsonl", "_summary.json"))
                eval_record = {
                    "run_key": experiment,
                    "seed": seed,
                    "stage": "eval",
                    "prompt_variant": prompt_mode,
                    "returncode": returncode,
                    "duration_seconds": duration,
                    "log_file": str(eval_log),
                    "summary_file": str(summary_path),
                    "checkpoint": str(checkpoint_path),
                    "command": eval_cmd,
                }
                records.append(eval_record)
                if returncode != 0:
                    raise RuntimeError(
                        f"Evaluation failed: experiment={experiment} prompt={prompt_mode} seed={seed} log={eval_log}"
                    )
                by_run_prompt_seed.setdefault(experiment, {}).setdefault(prompt_mode, {})[seed] = load_json(summary_path)

    aggregated: Dict[str, Dict[str, Any]] = {}
    experiment_contracts: Dict[str, Dict[str, Any]] = {}
    for run_key, prompt_map in by_run_prompt_seed.items():
        aggregated[run_key] = {}
        experiment_contracts[run_key] = {}
        for prompt_mode, seed_map in prompt_map.items():
            aggregated[run_key][prompt_mode] = {
                "n_success_runs": len(seed_map),
                "metrics": aggregate_mode_results(seed_map),
            }
            for _, summary in sorted(seed_map.items()):
                experiment_contracts[run_key][prompt_mode] = extract_experiment_contract(summary)
                break

    significance, significance_meta = compute_significance(
        by_run_prompt_seed=by_run_prompt_seed,
        prompt_modes=prompt_modes,
        seeds=seeds,
        primary_metric=args.report_primary_metric,
    )

    suite_summary = {
        "timestamp": timestamp,
        "config": args.config,
        "protocol": args.protocol,
        "dataset": dataset_name,
        "split": args.split,
        "train_num_samples": args.train_num_samples,
        "eval_num_samples": args.eval_num_samples,
        "base_model": args.base_model,
        "experiments": experiments,
        "seeds": seeds,
        "prompt_modes": prompt_modes,
        "fewshot_num_examples": args.fewshot_num_examples,
        "primary_metric": args.report_primary_metric,
        "records": records,
        "experiment_contracts": experiment_contracts,
        "aggregated": aggregated,
        "significance": significance,
        **significance_meta,
    }

    summary_json = suite_dir / "suite_summary.json"
    summary_json.write_text(json.dumps(suite_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        f"# Mini Matrix ({timestamp})",
        "",
        f"- dataset: `{dataset_name}`",
        f"- split: `{args.split}`",
        f"- train_num_samples: `{args.train_num_samples}`",
        f"- eval_num_samples: `{args.eval_num_samples}`",
        f"- primary_metric: `{args.report_primary_metric}`",
        "",
    ]
    for run_key, prompt_map in aggregated.items():
        md_lines.append(f"## {run_key}")
        for prompt_mode, payload in prompt_map.items():
            md_lines.append(f"### {prompt_mode}")
            md_lines.append("| metric | mean | std | ci95_low | ci95_high | n_runs |")
            md_lines.append("|---|---:|---:|---:|---:|---:|")
            for metric, row in payload.get("metrics", {}).items():
                ci95 = row.get("ci95", [0.0, 0.0])
                md_lines.append(
                    f"| {metric} | {row['mean']:.6f} | {row['std']:.6f} | {ci95[0]:.6f} | {ci95[1]:.6f} | {int(row['n_runs'])} |"
                )
            md_lines.append("")
    if significance:
        md_lines.append("## Significance")
        md_lines.append("| comparison | metric | p_value | observed_mean_diff | n_pairs | method |")
        md_lines.append("|---|---:|---:|---:|---:|---|")
        for comparison, metrics in significance.items():
            for metric, stat in metrics.items():
                md_lines.append(
                    f"| {comparison} | {metric} | {stat['p_value']:.6f} | {stat['observed_mean_diff']:.6f} | "
                    f"{int(stat['n_pairs'])} | {stat['method']} |"
                )
        md_lines.append("")
    elif significance_meta.get("significance_status", "").startswith("skipped_"):
        md_lines.append("## Significance")
        md_lines.append(f"- status: `{significance_meta['significance_status']}`")
        md_lines.append(f"- reason: {significance_meta['significance_skipped_reason']}")
        md_lines.append("")
    summary_md = suite_dir / "suite_summary.md"
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[DONE] {summary_json}")
    print(f"[DONE] {summary_md}")


if __name__ == "__main__":
    main()
