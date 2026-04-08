#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run local reproducibility suite for base/full/A2 checkpoint comparisons.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

from oglans.config import ConfigManager
from oglans.utils.pathing import (
    infer_dataset_name_from_config as infer_dataset_name_from_loaded_config,
    infer_eval_root_from_config,
)


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
    "primary_metric": "doc_role_micro_f1",
    "canonical_metric_mode": "analysis_only",
    "evaluation": {
        "split": "dev",
        "seeds": [3407, 3408, 3409],
    },
}

REQUIRED_SHARED_META = (
    "protocol_hash_sha256",
    "prompt_builder_version",
    "parser_version",
    "normalization_version",
)
REQUIRED_ADAPTER_META = REQUIRED_SHARED_META + ("checkpoint",)
TARGET_METRICS = (
    "doc_role_micro_f1",
    "doc_instance_micro_f1",
    "doc_combination_micro_f1",
    "doc_event_type_micro_f1",
    "strict_f1",
    "relaxed_f1",
    "type_f1",
    "strict_precision",
    "strict_recall",
    "schema_compliance_rate",
    "hallucination_rate",
)


# Keep this import-safe for tests that load the script via exec_module without
# registering it in sys.modules first.
class RunRecord(NamedTuple):
    run_key: str
    seed: int
    command: List[str]
    output_file: str
    summary_file: str
    run_manifest_file: str
    returncode: int
    duration_seconds: float
    ok: bool
    error: Optional[str] = None


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def deep_merge(base: Dict, override: Dict) -> Dict:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_protocol(path: str) -> Dict:
    protocol_path = Path(path)
    if not protocol_path.is_absolute():
        protocol_path = (PROJECT_ROOT / protocol_path).resolve()
    if not protocol_path.exists():
        return dict(DEFAULT_PROTOCOL)
    with protocol_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Protocol must be a dict: {protocol_path}")
    return deep_merge(DEFAULT_PROTOCOL, payload)


def parse_seeds(text: str) -> List[int]:
    seeds: List[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            seeds.append(int(token))
    if not seeds:
        raise ValueError("No valid seeds parsed.")
    return seeds


def parse_checkpoint_mapping(text: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not text:
        return mapping
    for item in str(text).split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid checkpoint mapping: {item}")
        key, value = item.split("=", 1)
        run_key = key.strip().lower()
        checkpoint = value.strip()
        if not run_key or not checkpoint:
            raise ValueError(f"Invalid checkpoint mapping: {item}")
        if run_key == "base":
            raise ValueError("checkpoint mapping must not redefine base run")
        mapping[run_key] = checkpoint
    return mapping


def build_eval_command(
    *,
    evaluate_path: Path,
    config: str,
    protocol: str,
    role_alias_map: str,
    run_key: str,
    split: str,
    seed: int,
    output_file: Path,
    batch_size: int,
    canonical_metric_mode: str,
    report_primary_metric: str,
    model_name_or_path: Optional[str],
    checkpoint_path: Optional[str],
    prompt_variant: Optional[str] = None,
    fewshot_num_examples: Optional[int] = None,
) -> List[str]:
    cmd = [
        sys.executable,
        str(evaluate_path),
        "--config",
        config,
        "--protocol",
        protocol,
        "--role_alias_map",
        role_alias_map,
        "--split",
        split,
        "--seed",
        str(seed),
        "--batch_size",
        str(batch_size),
        "--output_file",
        str(output_file),
        "--canonical_metric_mode",
        canonical_metric_mode,
        "--report_primary_metric",
        report_primary_metric,
    ]
    if model_name_or_path:
        cmd.extend(["--model_name_or_path", model_name_or_path])
    if prompt_variant:
        cmd.extend(["--prompt_variant", prompt_variant])
    if fewshot_num_examples is not None and str(prompt_variant).lower() == "fewshot":
        cmd.extend(["--fewshot_num_examples", str(fewshot_num_examples)])
    if run_key == "base":
        cmd.append("--base_only")
    else:
        if not checkpoint_path:
            raise ValueError(f"checkpoint_path is required for adapter run: {run_key}")
        cmd.extend(["--checkpoint", checkpoint_path])
    return cmd


def _missing_metadata(meta: Dict, required_keys: Sequence[str]) -> List[str]:
    missing = []
    for key in required_keys:
        value = meta.get(key)
        if value in (None, "", []):
            missing.append(key)
    return missing


def validate_eval_artifacts(
    *,
    summary_file: Path,
    run_manifest_file: Path,
    run_key: str,
) -> Dict:
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    if not run_manifest_file.exists():
        raise FileNotFoundError(f"Run manifest not found: {run_manifest_file}")

    summary = load_json(summary_file)
    manifest = load_json(run_manifest_file)
    summary_meta = summary.get("meta", {}) or {}
    manifest_meta = manifest.get("meta", {}) or {}
    required = REQUIRED_SHARED_META if run_key == "base" else REQUIRED_ADAPTER_META

    missing = sorted(
        set(_missing_metadata(summary_meta, required) + _missing_metadata(manifest_meta, required))
    )
    if missing:
        raise ValueError(f"missing required metadata: {', '.join(missing)}")

    for key in REQUIRED_SHARED_META:
        if summary_meta.get(key) != manifest_meta.get(key):
            raise ValueError(f"metadata mismatch between summary and manifest: {key}")
    if run_key != "base" and summary_meta.get("checkpoint") != manifest_meta.get("checkpoint"):
        raise ValueError("metadata mismatch between summary and manifest: checkpoint")

    validated = dict(summary)
    validated["manifest_meta"] = manifest_meta
    return validated


def ensure_complete_seed_coverage(
    *,
    records: Sequence[RunRecord],
    run_keys: Sequence[str],
    seeds: Sequence[int],
) -> None:
    expected = {(str(run_key), int(seed)) for run_key in run_keys for seed in seeds}
    successful = {(record.run_key, int(record.seed)) for record in records if record.ok}
    missing = sorted(expected - successful)
    failed = sorted((record.run_key, int(record.seed)) for record in records if not record.ok)
    if missing or failed:
        details = []
        if failed:
            details.append(f"failed={failed}")
        if missing:
            details.append(f"missing={missing}")
        raise ValueError(f"incomplete seed coverage: {'; '.join(details)}")


def infer_dataset_name_from_config(config_path: str) -> str:
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()
    try:
        cfg = ConfigManager.load_config(str(cfg_path))
    except Exception:
        return "DuEE-Fin"
    return infer_dataset_name_from_loaded_config(cfg) or "DuEE-Fin"


def infer_eval_academic_root_from_config(config_path: str, dataset_name: str) -> Path:
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = (PROJECT_ROOT / cfg_path).resolve()
    try:
        cfg = ConfigManager.load_config(str(cfg_path))
    except Exception:
        return PROJECT_ROOT / "logs" / dataset_name / "eval_academic"
    return PROJECT_ROOT / infer_eval_root_from_config(cfg, dataset_name, eval_task="eval_academic")


def run_one(cmd: List[str], cwd: Path) -> Tuple[int, float, Optional[str]]:
    start = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    except Exception as exc:
        return 1, time.time() - start, str(exc)
    duration = time.time() - start
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        stdout = (proc.stdout or "").strip()
        return proc.returncode, duration, stderr or stdout[-2000:] or "unknown error"
    return 0, duration, None


def markdown_table(agg: Dict[str, Dict], metrics: Sequence[str]) -> str:
    lines = [
        "| metric | mean | std | ci95_low | ci95_high | n_runs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        row = agg.get(metric)
        if not row:
            continue
        ci95 = row.get("ci95", [0.0, 0.0])
        lines.append(
            f"| {metric} | {row['mean']:.6f} | {row['std']:.6f} | {ci95[0]:.6f} | {ci95[1]:.6f} | {int(row['n_runs'])} |"
        )
    return "\n".join(lines)


def _build_run_metrics(summary: Dict, seed: int) -> Dict[str, float]:
    metrics = summary.get("metrics", {}) or {}
    row = {metric: float(metrics[metric]) for metric in TARGET_METRICS if metric in metrics}
    row["seed"] = float(seed)
    return row


def _compute_significance(
    validated_by_run: Dict[str, Dict[int, Dict]],
    report_primary_metric: str,
    expected_seeds: Sequence[int],
) -> Dict[str, Dict[str, Dict]]:
    result: Dict[str, Dict[str, Dict]] = {}
    a2_alias = next((key for key in validated_by_run if key in {"a2", "a2_no_scv", "a2-noscv"}), None)
    comparison_pairs = [("base", "full")]
    if a2_alias is not None:
        comparison_pairs.append(("full", a2_alias))
    metric_order = [report_primary_metric] + [
        m
        for m in (
            "doc_instance_micro_f1",
            "doc_combination_micro_f1",
            "doc_event_type_micro_f1",
            "strict_f1",
            "type_f1",
            "schema_compliance_rate",
        )
        if m != report_primary_metric
    ]

    for baseline_key, improved_key in comparison_pairs:
        if baseline_key not in validated_by_run or improved_key not in validated_by_run:
            continue
        common_seeds = sorted(set(validated_by_run[baseline_key].keys()) & set(validated_by_run[improved_key].keys()))
        if common_seeds != sorted(int(seed) for seed in expected_seeds):
            raise ValueError(
                f"incomplete seed coverage for significance: {baseline_key} vs {improved_key}, "
                f"expected={list(expected_seeds)}, got={common_seeds}"
            )
        pair_key = f"{baseline_key}_vs_{improved_key}"
        result[pair_key] = {}
        for metric in metric_order:
            baseline_scores = []
            improved_scores = []
            for seed in common_seeds:
                baseline_value = validated_by_run[baseline_key][seed].get("metrics", {}).get(metric)
                improved_value = validated_by_run[improved_key][seed].get("metrics", {}).get(metric)
                if baseline_value is None or improved_value is None:
                    continue
                baseline_scores.append(float(baseline_value))
                improved_scores.append(float(improved_value))
            if baseline_scores and improved_scores:
                result[pair_key][metric] = paired_permutation_pvalue(
                    baseline_scores=baseline_scores,
                    improved_scores=improved_scores,
                    seed=3407,
                )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local reproducibility suite for base/full/A2.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--protocol", type=str, default="configs/eval_protocol.yaml")
    parser.add_argument("--base-model", dest="base_model", type=str, required=True)
    parser.add_argument("--checkpoints", type=str, default="")
    parser.add_argument("--seeds", type=str, default="3407,3408,3409")
    parser.add_argument("--split", type=str, default=None, choices=["train", "dev", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--role_alias_map", type=str, default="configs/role_aliases_duee_fin.yaml")
    parser.add_argument("--prompt_variant", type=str, default=None, choices=["zeroshot", "fewshot"])
    parser.add_argument("--fewshot_num_examples", type=int, default=3)
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
    primary_metric = str(args.report_primary_metric or protocol.get("primary_metric", "doc_role_micro_f1"))
    canonical_metric_mode = str(args.canonical_metric_mode or protocol.get("canonical_metric_mode", "analysis_only"))
    split = str(args.split or protocol_split)
    seeds = parse_seeds(args.seeds)
    if split != protocol_split:
        raise ValueError(f"Split mismatch with protocol: args.split={split}, protocol.split={protocol_split}")
    if seeds != protocol_seeds:
        raise ValueError(f"Seeds mismatch with protocol: args.seeds={seeds}, protocol.seeds={protocol_seeds}")

    checkpoint_map = parse_checkpoint_mapping(args.checkpoints)
    run_targets = {"base": None}
    run_targets.update(checkpoint_map)
    dataset_name = infer_dataset_name_from_config(args.config)
    eval_root = infer_eval_academic_root_from_config(args.config, dataset_name)
    ts = time.strftime("%Y%m%d_%H%M%S")
    suite_dir = Path(args.output_dir) if args.output_dir else (eval_root / f"local_repro_suite_{ts}")
    suite_dir.mkdir(parents=True, exist_ok=True)

    evaluate_path = PROJECT_ROOT / "evaluate.py"
    if not evaluate_path.exists():
        raise FileNotFoundError(f"Cannot find evaluate.py at {evaluate_path}")

    records: List[RunRecord] = []
    validated_by_run: Dict[str, Dict[int, Dict]] = {}
    shared_reference_meta: Optional[Dict[str, object]] = None

    for run_key, checkpoint_path in run_targets.items():
        for seed in seeds:
            run_dir = suite_dir / f"{run_key}_seed{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            output_file = run_dir / "eval_results.jsonl"
            summary_file = run_dir / "eval_results_summary.json"
            run_manifest_file = run_dir / "run_manifest.json"
            cmd = build_eval_command(
                evaluate_path=evaluate_path,
                config=args.config,
                protocol=args.protocol,
                role_alias_map=args.role_alias_map,
                run_key=run_key,
                split=split,
                seed=seed,
                output_file=output_file,
                batch_size=args.batch_size,
                canonical_metric_mode=canonical_metric_mode,
                report_primary_metric=primary_metric,
                model_name_or_path=args.base_model,
                checkpoint_path=checkpoint_path,
                prompt_variant=args.prompt_variant,
                fewshot_num_examples=args.fewshot_num_examples,
            )

            print(f"[RUN] run={run_key} seed={seed}")
            returncode, duration, error = run_one(cmd, PROJECT_ROOT)
            record = RunRecord(
                run_key=run_key,
                seed=seed,
                command=cmd,
                output_file=str(output_file),
                summary_file=str(summary_file),
                run_manifest_file=str(run_manifest_file),
                returncode=returncode,
                duration_seconds=duration,
                ok=(returncode == 0),
                error=error,
            )
            records.append(record)
            if returncode != 0:
                print(f"[FAIL] run={run_key} seed={seed} rc={returncode}")
                continue

            validated = validate_eval_artifacts(
                summary_file=summary_file,
                run_manifest_file=run_manifest_file,
                run_key=run_key,
            )
            validated_by_run.setdefault(run_key, {})[seed] = validated

            current_shared_meta = {key: validated["meta"][key] for key in REQUIRED_SHARED_META}
            if shared_reference_meta is None:
                shared_reference_meta = current_shared_meta
            elif current_shared_meta != shared_reference_meta:
                raise ValueError(
                    f"metadata mismatch across runs for {run_key}_seed{seed}: {current_shared_meta} != {shared_reference_meta}"
                )
            print(f"[OK] run={run_key} seed={seed} ({duration:.1f}s)")

    ensure_complete_seed_coverage(
        records=records,
        run_keys=list(run_targets.keys()),
        seeds=seeds,
    )

    aggregated: Dict[str, Dict[str, object]] = {}
    for run_key, per_seed in validated_by_run.items():
        metric_rows = [_build_run_metrics(summary, seed) for seed, summary in sorted(per_seed.items())]
        metric_keys = [metric for metric in TARGET_METRICS if metric_rows and all(metric in row for row in metric_rows)]
        first_summary = next(iter(per_seed.values())) if per_seed else None
        aggregated[run_key] = {
            "n_success_runs": len(metric_rows),
            "metrics": aggregate_runs(metric_rows, metric_keys),
            "shared_meta": (
                {key: first_summary["meta"][key] for key in REQUIRED_SHARED_META}
                if first_summary is not None
                else {}
            ),
        }
        if run_key != "base" and first_summary is not None:
            aggregated[run_key]["checkpoint"] = first_summary["meta"]["checkpoint"]

    significance = _compute_significance(validated_by_run, primary_metric, seeds)
    suite_summary = {
        "timestamp": ts,
        "config": args.config,
        "protocol_path": str((PROJECT_ROOT / args.protocol).resolve()) if not Path(args.protocol).is_absolute() else args.protocol,
        "dataset": dataset_name,
        "split": split,
        "seeds": seeds,
        "base_model": args.base_model,
        "checkpoints": checkpoint_map,
        "primary_metric": primary_metric,
        "canonical_metric_mode": canonical_metric_mode,
        "prompt_variant": args.prompt_variant or "zeroshot",
        "fewshot_num_examples": args.fewshot_num_examples if args.prompt_variant == "fewshot" else 0,
        "runs": [record._asdict() for record in records],
        "aggregated": aggregated,
        "significance": significance,
        "shared_reference_meta": shared_reference_meta or {},
    }

    suite_json = suite_dir / "suite_summary.json"
    with suite_json.open("w", encoding="utf-8") as f:
        json.dump(suite_summary, f, ensure_ascii=False, indent=2)

    md_lines = [f"# Local Reproducibility Suite ({ts})", ""]
    for run_key in run_targets.keys():
        md_lines.append(f"## {run_key}")
        md_lines.append(markdown_table(aggregated.get(run_key, {}).get("metrics", {}), TARGET_METRICS))
        md_lines.append("")
    if significance:
        md_lines.append("## Significance")
        md_lines.append("| comparison | metric | p_value | observed_mean_diff | n_pairs | method |")
        md_lines.append("|---|---|---:|---:|---:|---|")
        for pair_key, metrics in significance.items():
            for metric, stat in metrics.items():
                md_lines.append(
                    f"| {pair_key} | {metric} | {stat['p_value']:.6f} | {stat['observed_mean_diff']:.6f} | "
                    f"{int(stat['n_pairs'])} | {stat['method']} |"
                )
        md_lines.append("")

    suite_md = suite_dir / "suite_summary.md"
    suite_md.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"[DONE] {suite_json}")
    print(f"[DONE] {suite_md}")


if __name__ == "__main__":
    main()
