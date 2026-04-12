#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run reproducible API evaluation suite (multi-seed, zero/few-shot, significance test).
"""

import argparse
import importlib.util
import json
import subprocess
import sys
import time
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from oglans.config import ConfigManager
from oglans.utils.pathing import (
    infer_dataset_name_from_config as infer_dataset_name_from_loaded_config,
    infer_eval_root_from_config,
)
from oglans.utils.compare_contract import extract_compare_contract, validate_compare_contract_match

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACADEMIC_EVAL_PATH = PROJECT_ROOT / "src" / "oglans" / "utils" / "academic_eval.py"
spec = importlib.util.spec_from_file_location("academic_eval", str(ACADEMIC_EVAL_PATH))
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load academic_eval from {ACADEMIC_EVAL_PATH}")
academic_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(academic_eval)
aggregate_runs = academic_eval.aggregate_runs
append_efficiency_metrics = academic_eval.append_efficiency_metrics
build_significance_metadata = academic_eval.build_significance_metadata
CORE_DIAGNOSTIC_REPORT_METRICS = academic_eval.CORE_DIAGNOSTIC_REPORT_METRICS
ACADEMIC_MAIN_TABLE_METRICS = academic_eval.ACADEMIC_MAIN_TABLE_METRICS
EFFICIENCY_REPORT_METRICS = academic_eval.EFFICIENCY_REPORT_METRICS
API_SUITE_REPORT_METRICS = academic_eval.API_SUITE_REPORT_METRICS
COST_REPORT_METRICS = academic_eval.COST_REPORT_METRICS
extract_report_metrics = academic_eval.extract_report_metrics
MIN_SIGNIFICANCE_PAIRS = academic_eval.MIN_SIGNIFICANCE_PAIRS
paired_permutation_pvalue = academic_eval.paired_permutation_pvalue

DEFAULT_PROTOCOL = {
    "version": "1.0",
    "primary_metric": "doc_role_micro_f1",
    "canonical_metric_mode": "analysis_only",
    "evaluation": {
        "split": "dev",
        "seeds": [3407, 3408, 3409],
        "bootstrap_samples": 1000,
        "concurrency": 8,
        "significance": "paired_permutation",
    },
    "metrics": {
        "version": "2.0",
        "report_level": "core_plus_diagnostics",
        "cot": {
            "enabled": False,
            "mode": "strict_span",
            "require_thought_block": False,
        },
        "relaxed": {
            "match_mode": "include_or_char_overlap",
            "char_overlap_threshold": 0.5,
            "span_iou_threshold": 0.5,
        },
        "hallucination": {
            "match_mode": "normalized_substring",
        },
        "schema": {
            "mode": "schema_strict",
        },
    },
}


@dataclass
class RunRecord:
    mode: str
    seed: int
    command: List[str]
    output_file: str
    summary_file: str
    returncode: int
    duration_seconds: float
    ok: bool
    error: Optional[str] = None


def parse_seeds(text: str) -> List[int]:
    seeds = []
    for token in text.split(","):
        token = token.strip()
        if token:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("No valid seeds parsed.")
    return seeds


def parse_modes(text: str) -> List[str]:
    raw = [x.strip().lower() for x in text.split(",") if x.strip()]
    valid = {"zeroshot", "fewshot"}
    for m in raw:
        if m not in valid:
            raise ValueError(f"Invalid mode: {m}. Valid modes: zeroshot,fewshot")
    if not raw:
        raise ValueError("No valid modes parsed.")
    return raw


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_metric_row(summary: Dict, target_metrics: List[str], seed: int) -> Dict[str, float]:
    row = extract_report_metrics(
        summary,
        required_metrics=target_metrics,
        optional_metrics=COST_REPORT_METRICS,
    )
    row = append_efficiency_metrics(row)
    row["seed"] = float(seed)
    return row


def validate_mode_contracts(by_mode_seed: Dict[str, Dict[int, Dict]]) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for mode, seed_map in by_mode_seed.items():
        compare_blocks = [extract_compare_contract(summary) for _, summary in sorted(seed_map.items())]
        hashes[mode] = validate_compare_contract_match(compare_blocks)
    return hashes


def deep_merge(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_protocol(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        return dict(DEFAULT_PROTOCOL)
    with p.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Protocol must be a dict: {path}")
    return deep_merge(DEFAULT_PROTOCOL, payload)


def compute_significance(
    by_mode_seed: Dict[str, Dict[int, Dict]],
    report_primary_metric: str,
    expected_seeds: List[int],
) -> Tuple[Dict[str, Dict], Dict[str, object]]:
    significance: Dict[str, Dict] = {}
    common_seeds = sorted(set(by_mode_seed.get("zeroshot", {}).keys()) & set(by_mode_seed.get("fewshot", {}).keys()))
    if common_seeds != sorted(int(seed) for seed in expected_seeds):
        raise ValueError(
            "incomplete seed coverage for significance: "
            f"expected={sorted(int(seed) for seed in expected_seeds)}, got={common_seeds}"
        )
    metadata = build_significance_metadata([len(common_seeds)])
    if len(common_seeds) < MIN_SIGNIFICANCE_PAIRS:
        return significance, metadata

    sig_metrics = [report_primary_metric] + [
        m
        for m in [
            "doc_instance_micro_f1",
            "doc_combination_micro_f1",
            "doc_event_type_micro_f1",
            "single_event_doc_role_micro_f1",
            "multi_event_doc_role_micro_f1",
            "strict_f1",
            "relaxed_f1",
            "type_f1",
        ]
        if m != report_primary_metric
    ]
    for metric in sig_metrics:
        baseline_scores = []
        improved_scores = []
        for seed in common_seeds:
            baseline_value = extract_report_metrics(
                by_mode_seed["zeroshot"][seed],
                required_metrics=(metric,),
            )[metric]
            improved_value = extract_report_metrics(
                by_mode_seed["fewshot"][seed],
                required_metrics=(metric,),
            )[metric]
            baseline_scores.append(float(baseline_value))
            improved_scores.append(float(improved_value))
        significance[metric] = paired_permutation_pvalue(
            baseline_scores=baseline_scores,
            improved_scores=improved_scores,
            seed=3407,
        )
    return significance, metadata


def build_cmd(
    evaluate_api_path: Path,
    config: str,
    protocol: str,
    mode: str,
    split: str,
    seed: int,
    output_file: Path,
    summary_file: Path,
    concurrency: int,
    json_mode: str,
    model: Optional[str],
    base_url: Optional[str],
    num_samples: Optional[int],
    bootstrap_samples: Optional[int],
    role_alias_map: str,
    canonical_metric_mode: str,
    report_primary_metric: str,
    fewshot_num_examples: Optional[int],
    stage_mode: str = "single_pass",
    fewshot_selection_mode: str = "dynamic",
    fewshot_pool_split: str = "train_fit",
    train_tune_ratio: Optional[float] = None,
    research_split_manifest: Optional[str] = None,
) -> List[str]:
    cmd = [
        sys.executable,
        str(evaluate_api_path),
        "--config", config,
        "--protocol", protocol,
        "--split", split,
        "--seed", str(seed),
        "--concurrency", str(concurrency),
        "--json_mode", json_mode,
        "--output_file", str(output_file),
        "--summary_file", str(summary_file),
        "--role_alias_map", role_alias_map,
        "--canonical_metric_mode", canonical_metric_mode,
        "--report_primary_metric", report_primary_metric,
        "--stage_mode", stage_mode,
        "--fewshot_selection_mode", fewshot_selection_mode,
        "--fewshot_pool_split", fewshot_pool_split,
    ]
    if model:
        cmd.extend(["--model", model])
    if base_url:
        cmd.extend(["--base_url", base_url])
    if num_samples is not None:
        cmd.extend(["--num_samples", str(num_samples)])
    if bootstrap_samples is not None:
        cmd.extend(["--bootstrap_samples", str(bootstrap_samples)])
    if train_tune_ratio is not None:
        cmd.extend(["--train_tune_ratio", str(train_tune_ratio)])
    if research_split_manifest:
        cmd.extend(["--research_split_manifest", research_split_manifest])
    if mode == "fewshot":
        cmd.append("--use_fewshot")
        if fewshot_num_examples is not None:
            cmd.extend(["--fewshot_num_examples", str(fewshot_num_examples)])
    return cmd


def run_one(cmd: List[str], cwd: Path) -> Tuple[int, float, Optional[str]]:
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    except Exception as e:
        return 1, time.time() - t0, str(e)
    duration = time.time() - t0
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        err = stderr if stderr else stdout[-1000:]
        return proc.returncode, duration, err
    return 0, duration, None


def markdown_table(agg: Dict[str, Dict], metrics: List[str]) -> str:
    lines = [
        "| metric | mean | std | ci95_low | ci95_high | n_runs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m in metrics:
        if m not in agg:
            continue
        row = agg[m]
        ci = row.get("ci95", [0.0, 0.0])
        lines.append(
            f"| {m} | {row['mean']:.6f} | {row['std']:.6f} | {ci[0]:.6f} | {ci[1]:.6f} | {int(row['n_runs'])} |"
        )
    return "\n".join(lines)


def infer_dataset_name_from_config(config_path: str) -> str:
    """Infer dataset name from config.yaml to avoid hardcoded output paths."""
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()
    cfg = ConfigManager._load_config_file(str(cfg_path))
    dataset_name = infer_dataset_name_from_loaded_config(cfg)
    if not dataset_name:
        raise ValueError(f"Unable to infer dataset name from config: {cfg_path}")
    return dataset_name


def infer_eval_api_root_from_config(config_path: str, dataset_name: str) -> Path:
    """
    Infer eval_api root directory from config.project paths.
    Debug configs like logs/debug/checkpoints will map to logs/debug/eval_api.
    """
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()
    cfg = ConfigManager._load_config_file(str(cfg_path))
    return PROJECT_ROOT / infer_eval_root_from_config(cfg, dataset_name, eval_task="eval_api")


def main():
    parser = argparse.ArgumentParser(description="Run reproducible API evaluation suite.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--protocol", type=str, default="configs/eval_protocol.yaml")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test", "train"])
    parser.add_argument("--seeds", type=str, default="3407,3408,3409")
    parser.add_argument("--modes", type=str, default="zeroshot,fewshot")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--json_mode", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--bootstrap_samples", type=int, default=None)
    parser.add_argument("--fewshot_num_examples", type=int, default=None)
    parser.add_argument("--stage_mode", type=str, default="single_pass", choices=["single_pass", "two_stage"])
    parser.add_argument("--fewshot_selection_mode", type=str, default="dynamic", choices=["static", "dynamic"])
    parser.add_argument("--fewshot_pool_split", type=str, default="train_fit", choices=["train", "train_fit"])
    parser.add_argument("--train_tune_ratio", type=float, default=None)
    parser.add_argument("--research_split_manifest", type=str, default=None)
    parser.add_argument("--role_alias_map", type=str, default="configs/role_aliases_duee_fin.yaml")
    parser.add_argument(
        "--canonical_metric_mode",
        type=str,
        default=None,
        choices=["off", "analysis_only", "apply_for_aux_metric"],
    )
    parser.add_argument("--report_primary_metric", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    protocol_eval = protocol.get("evaluation", {}) if isinstance(protocol, dict) else {}
    protocol_split = str(protocol_eval.get("split", "dev"))
    protocol_seeds = [int(x) for x in protocol_eval.get("seeds", [3407, 3408, 3409])]
    protocol_primary = str(protocol.get("primary_metric", "doc_role_micro_f1"))
    protocol_canonical_mode = str(protocol.get("canonical_metric_mode", "analysis_only"))

    if args.split != protocol_split:
        raise ValueError(
            f"Split mismatch with protocol: args.split={args.split}, protocol.split={protocol_split}"
        )

    seeds = parse_seeds(args.seeds)
    if seeds != protocol_seeds:
        raise ValueError(
            f"Seeds mismatch with protocol: args.seeds={seeds}, protocol.seeds={protocol_seeds}"
        )

    modes = parse_modes(args.modes)
    if args.concurrency is None:
        args.concurrency = int(protocol_eval.get("concurrency", 8))
    if args.bootstrap_samples is None:
        args.bootstrap_samples = int(protocol_eval.get("bootstrap_samples", 1000))
    if args.canonical_metric_mode is None:
        args.canonical_metric_mode = protocol_canonical_mode
    if args.report_primary_metric is None:
        args.report_primary_metric = protocol_primary
    dataset_name = infer_dataset_name_from_config(args.config)
    eval_api_root = infer_eval_api_root_from_config(args.config, dataset_name)

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (eval_api_root / f"repro_suite_{ts}")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    evaluate_api_path = PROJECT_ROOT / "evaluate_api.py"
    if not evaluate_api_path.exists():
        raise FileNotFoundError(f"Cannot find evaluate_api.py at {evaluate_api_path}")

    records: List[RunRecord] = []
    for mode in modes:
        for seed in seeds:
            tag = f"{mode}_seed{seed}"
            output_file = output_dir / f"{tag}.jsonl"
            summary_file = output_dir / f"{tag}_summary.json"
            cmd = build_cmd(
                evaluate_api_path=evaluate_api_path,
                config=args.config,
                protocol=args.protocol,
                mode=mode,
                split=args.split,
                seed=seed,
                output_file=output_file,
                summary_file=summary_file,
                concurrency=args.concurrency,
                json_mode=args.json_mode,
                model=args.model,
                base_url=args.base_url,
                num_samples=args.num_samples,
                bootstrap_samples=args.bootstrap_samples,
                role_alias_map=args.role_alias_map,
                canonical_metric_mode=args.canonical_metric_mode,
                report_primary_metric=args.report_primary_metric,
                fewshot_num_examples=args.fewshot_num_examples,
                stage_mode=args.stage_mode,
                fewshot_selection_mode=args.fewshot_selection_mode,
                fewshot_pool_split=args.fewshot_pool_split,
                train_tune_ratio=args.train_tune_ratio,
                research_split_manifest=args.research_split_manifest,
            )

            print(f"[RUN] mode={mode} seed={seed}")
            code, duration, err = run_one(cmd, PROJECT_ROOT)
            rec = RunRecord(
                mode=mode,
                seed=seed,
                command=cmd,
                output_file=str(output_file),
                summary_file=str(summary_file),
                returncode=code,
                duration_seconds=duration,
                ok=(code == 0),
                error=err,
            )
            records.append(rec)
            if code != 0:
                print(f"[FAIL] mode={mode} seed={seed} rc={code}")
            else:
                print(f"[OK] mode={mode} seed={seed} ({duration:.1f}s)")

    # Load successful summaries
    successful = [r for r in records if r.ok and Path(r.summary_file).exists()]
    by_mode_seed: Dict[str, Dict[int, Dict]] = {}
    for rec in successful:
        by_mode_seed.setdefault(rec.mode, {})[rec.seed] = load_json(Path(rec.summary_file))
    comparable_contract_hashes = validate_mode_contracts(by_mode_seed) if by_mode_seed else {}

    target_metrics = list(ACADEMIC_MAIN_TABLE_METRICS) + [
        metric for metric in API_SUITE_REPORT_METRICS if metric not in ACADEMIC_MAIN_TABLE_METRICS
    ]
    cost_metrics = list(COST_REPORT_METRICS) + list(EFFICIENCY_REPORT_METRICS)

    aggregated: Dict[str, Dict] = {}
    for mode, seed_map in by_mode_seed.items():
        run_metric_rows = []
        for seed, summary in sorted(seed_map.items()):
            run_metric_rows.append(build_metric_row(summary, target_metrics, seed))
        main_metric_keys = [k for k in target_metrics if all(k in r for r in run_metric_rows)]
        cost_metric_keys = [k for k in cost_metrics if all(k in r for r in run_metric_rows)]
        aggregated[mode] = {
            "n_success_runs": len(run_metric_rows),
            "metrics": aggregate_runs(run_metric_rows, main_metric_keys),
            "cost_metrics": aggregate_runs(run_metric_rows, cost_metric_keys),
        }

    significance, significance_meta = compute_significance(
        by_mode_seed,
        args.report_primary_metric,
        expected_seeds=seeds,
    )

    suite_summary = {
        "timestamp": ts,
        "config": args.config,
        "dataset": dataset_name,
        "split": args.split,
        "seeds": seeds,
        "modes": modes,
        "model": args.model,
        "base_url": args.base_url,
        "num_samples": args.num_samples,
        "concurrency": args.concurrency,
        "json_mode": args.json_mode,
        "bootstrap_samples": args.bootstrap_samples,
        "fewshot_num_examples": args.fewshot_num_examples,
        "stage_mode": args.stage_mode,
        "fewshot_selection_mode": args.fewshot_selection_mode,
        "fewshot_pool_split": args.fewshot_pool_split,
        "train_tune_ratio": args.train_tune_ratio,
        "research_split_manifest": args.research_split_manifest,
        "protocol_path": str((PROJECT_ROOT / args.protocol).resolve()) if not Path(args.protocol).is_absolute() else args.protocol,
        "protocol": protocol,
        "primary_metric": args.report_primary_metric,
        "canonical_metric_mode": args.canonical_metric_mode,
        "comparable_contract_hashes": comparable_contract_hashes,
        "runs": [asdict(r) for r in records],
        "aggregated": aggregated,
        "significance": significance,
        **significance_meta,
    }

    suite_json = output_dir / "suite_summary.json"
    with suite_json.open("w", encoding="utf-8") as f:
        json.dump(suite_summary, f, ensure_ascii=False, indent=2)

    md_lines = [f"# API Reproducibility Suite ({ts})", ""]
    for mode in modes:
        md_lines.append(f"## {mode}")
        mode_agg = aggregated.get(mode, {}).get("metrics", {})
        md_lines.append("### Main Metrics")
        md_lines.append(markdown_table(mode_agg, target_metrics))
        cost_agg = aggregated.get(mode, {}).get("cost_metrics", {})
        if cost_agg:
            md_lines.append("")
            md_lines.append("### Cost And Efficiency")
            md_lines.append(markdown_table(cost_agg, cost_metrics))
        md_lines.append("")
    if significance:
        md_lines.append("## Significance (Few-shot vs Zero-shot)")
        md_lines.append("| metric | p_value | observed_mean_diff | n_pairs | method |")
        md_lines.append("|---|---:|---:|---:|---|")
        for metric, stat in significance.items():
            md_lines.append(
                f"| {metric} | {stat['p_value']:.6f} | {stat['observed_mean_diff']:.6f} | "
                f"{int(stat['n_pairs'])} | {stat['method']} |"
            )
        md_lines.append("")
    elif significance_meta.get("significance_status", "").startswith("skipped_"):
        md_lines.append("## Significance (Few-shot vs Zero-shot)")
        md_lines.append(f"- status: `{significance_meta['significance_status']}`")
        md_lines.append(f"- reason: {significance_meta['significance_skipped_reason']}")
        md_lines.append("")

    suite_md = output_dir / "suite_summary.md"
    suite_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[DONE] {suite_json}")
    print(f"[DONE] {suite_md}")


if __name__ == "__main__":
    main()
