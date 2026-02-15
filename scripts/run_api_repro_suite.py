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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACADEMIC_EVAL_PATH = PROJECT_ROOT / "src" / "oglans" / "utils" / "academic_eval.py"
spec = importlib.util.spec_from_file_location("academic_eval", str(ACADEMIC_EVAL_PATH))
if spec is None or spec.loader is None:
    raise ImportError(f"Cannot load academic_eval from {ACADEMIC_EVAL_PATH}")
academic_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(academic_eval)
aggregate_runs = academic_eval.aggregate_runs
paired_permutation_pvalue = academic_eval.paired_permutation_pvalue

DEFAULT_PROTOCOL = {
    "version": "1.0",
    "primary_metric": "strict_f1",
    "canonical_metric_mode": "analysis_only",
    "evaluation": {
        "split": "dev",
        "seeds": [3407, 3408, 3409],
        "bootstrap_samples": 1000,
        "concurrency": 8,
        "significance": "paired_permutation",
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
    num_samples: Optional[int],
    bootstrap_samples: Optional[int],
    role_alias_map: str,
    canonical_metric_mode: str,
    report_primary_metric: str,
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
    ]
    if model:
        cmd.extend(["--model", model])
    if num_samples is not None:
        cmd.extend(["--num_samples", str(num_samples)])
    if bootstrap_samples is not None:
        cmd.extend(["--bootstrap_samples", str(bootstrap_samples)])
    if mode == "fewshot":
        cmd.append("--use_fewshot")
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
    try:
        cfg_path = Path(config_path)
        if not cfg_path.is_absolute():
            cfg_path = (PROJECT_ROOT / cfg_path).resolve()
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        return "DuEE-Fin"

    ds_cns = cfg.get("algorithms", {}).get("ds_cns", {})
    taxonomy_path = ds_cns.get("taxonomy_path")
    if taxonomy_path:
        p = Path(taxonomy_path)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        parent_name = p.parent.name
        if parent_name:
            return parent_name

    project = cfg.get("project", {})
    for key in ("dataset_cache_dir", "output_dir", "logging_dir"):
        raw = project.get(key)
        if not raw:
            continue
        p = Path(raw)
        if not p.is_absolute():
            p = (PROJECT_ROOT / p).resolve()
        parts = [x for x in p.parts if x]
        if not parts:
            continue
        if key == "dataset_cache_dir" and len(parts) >= 2:
            return parts[-2]
        base = parts[-1]
        if base in {"checkpoints", "tensorboard", "samples", "eval", "logs", "log", "train"} and len(parts) >= 2:
            base = parts[-2]
        if base:
            return base

    return "DuEE-Fin"


def main():
    parser = argparse.ArgumentParser(description="Run reproducible API evaluation suite.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--protocol", type=str, default="configs/eval_protocol.yaml")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test", "train"])
    parser.add_argument("--seeds", type=str, default="3407,3408,3409")
    parser.add_argument("--modes", type=str, default="zeroshot,fewshot")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--json_mode", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--bootstrap_samples", type=int, default=None)
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
    protocol_primary = str(protocol.get("primary_metric", "strict_f1"))
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

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (PROJECT_ROOT / "logs" / dataset_name / "eval" / f"repro_suite_{ts}")
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
                num_samples=args.num_samples,
                bootstrap_samples=args.bootstrap_samples,
                role_alias_map=args.role_alias_map,
                canonical_metric_mode=args.canonical_metric_mode,
                report_primary_metric=args.report_primary_metric,
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

    target_metrics = ["strict_f1", "relaxed_f1", "type_f1", "strict_precision", "strict_recall"]

    aggregated: Dict[str, Dict] = {}
    for mode, seed_map in by_mode_seed.items():
        run_metric_rows = []
        for seed, summary in sorted(seed_map.items()):
            metrics = summary.get("metrics", {})
            row = {k: float(metrics[k]) for k in target_metrics if k in metrics}
            if row:
                row["seed"] = seed
                run_metric_rows.append(row)
        aggregated[mode] = {
            "n_success_runs": len(run_metric_rows),
            "metrics": aggregate_runs(run_metric_rows, [k for k in target_metrics if all(k in r for r in run_metric_rows)]),
        }

    significance = {}
    if "zeroshot" in by_mode_seed and "fewshot" in by_mode_seed:
        common_seeds = sorted(set(by_mode_seed["zeroshot"].keys()) & set(by_mode_seed["fewshot"].keys()))
        sig_metrics = [args.report_primary_metric] + [
            m for m in ["strict_f1", "relaxed_f1", "type_f1"] if m != args.report_primary_metric
        ]
        for metric in sig_metrics:
            baseline_scores = []
            improved_scores = []
            for seed in common_seeds:
                b = by_mode_seed["zeroshot"][seed].get("metrics", {}).get(metric)
                i = by_mode_seed["fewshot"][seed].get("metrics", {}).get(metric)
                if b is None or i is None:
                    continue
                baseline_scores.append(float(b))
                improved_scores.append(float(i))
            if baseline_scores and improved_scores:
                significance[metric] = paired_permutation_pvalue(
                    baseline_scores=baseline_scores,
                    improved_scores=improved_scores,
                    seed=3407,
                )

    suite_summary = {
        "timestamp": ts,
        "config": args.config,
        "dataset": dataset_name,
        "split": args.split,
        "seeds": seeds,
        "modes": modes,
        "model": args.model,
        "num_samples": args.num_samples,
        "concurrency": args.concurrency,
        "json_mode": args.json_mode,
        "bootstrap_samples": args.bootstrap_samples,
        "protocol_path": str((PROJECT_ROOT / args.protocol).resolve()) if not Path(args.protocol).is_absolute() else args.protocol,
        "protocol": protocol,
        "primary_metric": args.report_primary_metric,
        "canonical_metric_mode": args.canonical_metric_mode,
        "runs": [asdict(r) for r in records],
        "aggregated": aggregated,
        "significance": significance,
    }

    suite_json = output_dir / "suite_summary.json"
    with suite_json.open("w", encoding="utf-8") as f:
        json.dump(suite_summary, f, ensure_ascii=False, indent=2)

    md_lines = [f"# API Reproducibility Suite ({ts})", ""]
    for mode in modes:
        md_lines.append(f"## {mode}")
        mode_agg = aggregated.get(mode, {}).get("metrics", {})
        md_lines.append(markdown_table(mode_agg, target_metrics))
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

    suite_md = output_dir / "suite_summary.md"
    suite_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[DONE] {suite_json}")
    print(f"[DONE] {suite_md}")


if __name__ == "__main__":
    main()
