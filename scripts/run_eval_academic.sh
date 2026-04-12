#!/usr/bin/env bash
set -euo pipefail

# OG-LANS Evaluation Runner
# - Deterministic multi-seed evaluation
# - Per-seed logs + metrics
# - Aggregated mean/std summary for paper reporting

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
PROJECT_ROOT="$ROOT_DIR"

# Runtime environment
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="offline"
export PYTHONUNBUFFERED="1"  # Ensure realtime logging without Python stdout buffering
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${PROJECT_ROOT}/models}"

resolve_python_bin() {
  if command -v python >/dev/null 2>&1; then
    echo "python"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
    return
  fi
  echo ""
}
PYTHON_BIN="$(resolve_python_bin)"
if [[ -z "$PYTHON_BIN" ]]; then
  echo "ERROR: neither python nor python3 found in PATH."
  exit 1
fi
mkdir -p "$MODELSCOPE_CACHE"

config_fields() {
  "$PYTHON_BIN" scripts/resolve_config_context.py --config "$CONFIG" --project-root "$PROJECT_ROOT" "$@"
}

CONFIG="configs/config.yaml"
PROTOCOL="configs/eval_protocol.yaml"
ROLE_ALIAS_MAP="configs/role_aliases_duee_fin.yaml"
CANONICAL_MODE="analysis_only"
PRIMARY_METRIC=""
CHECKPOINT=""
DATASET_NAME=""
SPLIT="dev"
BATCH_SIZE=4
EVAL_MODE="both"                # strict | relaxed | both
SEEDS="3407,3408,3409"
NUM_SAMPLES=""
USE_ONESHOT=0
PROMPT_VARIANT=""
FEWSHOT_NUM_EXAMPLES=""
DO_SAMPLE=0                     # keep 0 for academic deterministic eval
COT_EVAL_MODE="self_consistency"
PIPELINE_MODE="e2e"
OUT_DIR=""
TAG=""
SEED_POLICY="train_seed"        # train_seed | eval_seed
TAIL_ON_FAIL=120
ORIGINAL_CMD="bash $0 $*"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_eval_academic.sh --checkpoint <path> [options]

Required:
  --checkpoint <path>          LoRA checkpoint path

Options:
  --dataset-name <name>        Dataset name for output directory (optional)
  --config <path>              Config file (default: configs/config.yaml)
  --protocol <path>            Eval protocol path (default: configs/eval_protocol.yaml)
  --role-alias-map <path>      Role alias map path (default: configs/role_aliases_duee_fin.yaml)
  --canonical-mode <mode>      off|analysis_only|apply_for_aux_metric
  --primary-metric <name>      Primary metric key (optional, overrides protocol)
  --split <dev|test|train>     Dataset split (default: dev)
  --batch-size <int>           Inference batch size (default: 4)
  --eval-mode <strict|relaxed|both>
                               Metric mode (default: both)
  --seeds <csv>                Seeds, e.g. 3407,3408,3409
  --seed-policy <mode>         train_seed|eval_seed (default: train_seed)
  --num-samples <int>          Evaluate first N samples
  --use-oneshot                Enable one-shot prompt
  --prompt-variant <mode>      zeroshot|fewshot (optional)
  --fewshot-num-examples <int> Few-shot example count (optional)
  --do-sample                  Enable sampling decode (not recommended for papers)
  --cot-eval-mode <mode>       self_consistency|counterfactual
  --pipeline-mode <mode>       e2e|cat_lite
  --out-dir <path>             Output directory (default auto-generated)
  --tag <str>                  Optional run tag appended to output directory
  -h, --help                   Show help

Examples:
  # 推荐论文口径：每个训练种子对应一个checkpoint
  bash scripts/run_eval_academic.sh \
    --seed-policy train_seed \
    --seeds 3407,3408,3409 \
    --checkpoint logs/DuEE-Fin/checkpoints/exp_seed3407,logs/DuEE-Fin/checkpoints/exp_seed3408,logs/DuEE-Fin/checkpoints/exp_seed3409 \
    --split dev --eval-mode both

  # 探索口径：单checkpoint做eval-seed sweep（需显式允许）
  bash scripts/run_eval_academic.sh \
    --seed-policy eval_seed \
    --checkpoint logs/DuEE-Fin/checkpoints/exp1 \
    --num-samples 200 --seeds 3407,3408,3409
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="${2:-}"; shift 2 ;;
    --dataset-name) DATASET_NAME="${2:-}"; shift 2 ;;
    --config) CONFIG="${2:-}"; shift 2 ;;
    --protocol) PROTOCOL="${2:-}"; shift 2 ;;
    --role-alias-map) ROLE_ALIAS_MAP="${2:-}"; shift 2 ;;
    --canonical-mode) CANONICAL_MODE="${2:-}"; shift 2 ;;
    --primary-metric) PRIMARY_METRIC="${2:-}"; shift 2 ;;
    --split) SPLIT="${2:-}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:-}"; shift 2 ;;
    --eval-mode) EVAL_MODE="${2:-}"; shift 2 ;;
    --seeds) SEEDS="${2:-}"; shift 2 ;;
    --seed-policy) SEED_POLICY="${2:-}"; shift 2 ;;
    --num-samples) NUM_SAMPLES="${2:-}"; shift 2 ;;
    --use-oneshot) USE_ONESHOT=1; shift ;;
    --prompt-variant|--prompt_variant) PROMPT_VARIANT="${2:-}"; shift 2 ;;
    --fewshot-num-examples|--fewshot_num_examples) FEWSHOT_NUM_EXAMPLES="${2:-}"; shift 2 ;;
    --do-sample) DO_SAMPLE=1; shift ;;
    --cot-eval-mode) COT_EVAL_MODE="${2:-}"; shift 2 ;;
    --pipeline-mode) PIPELINE_MODE="${2:-}"; shift 2 ;;
    --out-dir) OUT_DIR="${2:-}"; shift 2 ;;
    --tag) TAG="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  echo "ERROR: --checkpoint is required."
  usage
  exit 1
fi
if [[ "$SEED_POLICY" != "train_seed" && "$SEED_POLICY" != "eval_seed" ]]; then
  echo "ERROR: --seed-policy must be one of: train_seed, eval_seed."
  exit 1
fi
if [[ "$COT_EVAL_MODE" != "self_consistency" && "$COT_EVAL_MODE" != "counterfactual" ]]; then
  echo "ERROR: --cot-eval-mode must be self_consistency or counterfactual."
  exit 1
fi
if [[ "$PIPELINE_MODE" != "e2e" && "$PIPELINE_MODE" != "cat_lite" ]]; then
  echo "ERROR: --pipeline-mode must be e2e or cat_lite."
  exit 1
fi

IFS=',' read -r -a CHECKPOINT_ARR_RAW <<< "$CHECKPOINT"
declare -a CHECKPOINT_ARR
for ckpt_raw in "${CHECKPOINT_ARR_RAW[@]}"; do
  ckpt="$(echo "$ckpt_raw" | xargs)"
  [[ -z "$ckpt" ]] && continue
  CHECKPOINT_ARR+=("$ckpt")
done
if [[ ${#CHECKPOINT_ARR[@]} -eq 0 ]]; then
  echo "ERROR: no valid checkpoint parsed from --checkpoint."
  exit 1
fi
PRIMARY_CHECKPOINT="${CHECKPOINT_ARR[0]}"

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG"
  exit 1
fi
if [[ ! -f "$PROTOCOL" ]]; then
  echo "ERROR: protocol not found: $PROTOCOL"
  exit 1
fi

for ckpt in "${CHECKPOINT_ARR[@]}"; do
  if [[ ! -e "$ckpt" ]]; then
    echo "ERROR: checkpoint path does not exist: $ckpt"
    exit 1
  fi
done
if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import yaml  # noqa: F401
PY
then
  echo "ERROR: missing dependency 'yaml' in current Python environment."
  exit 1
fi

IFS=',' read -r -a SEED_ARR <<< "$SEEDS"
declare -a CLEAN_SEEDS
for seed_raw in "${SEED_ARR[@]}"; do
  seed="$(echo "$seed_raw" | xargs)"
  [[ -z "$seed" ]] && continue
  CLEAN_SEEDS+=("$seed")
done
if [[ ${#CLEAN_SEEDS[@]} -eq 0 ]]; then
  echo "ERROR: no valid seeds parsed from --seeds."
  exit 1
fi
SEEDS="$(IFS=,; echo "${CLEAN_SEEDS[*]}")"
SEED_ARR=("${CLEAN_SEEDS[@]}")

if [[ "$SEED_POLICY" == "train_seed" ]]; then
  if [[ ${#SEED_ARR[@]} -gt 1 && ${#CHECKPOINT_ARR[@]} -eq 1 ]]; then
    echo "ERROR: seed_policy=train_seed requires one checkpoint per seed."
    echo "  seeds: ${SEED_ARR[*]}"
    echo "  checkpoints parsed: ${#CHECKPOINT_ARR[@]}"
    echo "Hint: pass comma-separated checkpoints matching seeds, or use --seed-policy eval_seed."
    exit 1
  fi
  if [[ ${#CHECKPOINT_ARR[@]} -ne ${#SEED_ARR[@]} ]]; then
    echo "ERROR: seed_policy=train_seed requires len(checkpoints) == len(seeds)."
    echo "  seeds: ${#SEED_ARR[@]}, checkpoints: ${#CHECKPOINT_ARR[@]}"
    exit 1
  fi
else
  if [[ ${#CHECKPOINT_ARR[@]} -ne 1 ]]; then
    echo "ERROR: seed_policy=eval_seed supports exactly one checkpoint."
    exit 1
  fi
  if [[ "$DO_SAMPLE" -eq 0 && ${#SEED_ARR[@]} -gt 1 ]]; then
    echo "ERROR: eval_seed + deterministic decode on one checkpoint has weak statistical meaning."
    echo "Use --seed-policy train_seed with one checkpoint per seed."
    exit 1
  fi
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
DATASET="$DATASET_NAME"
if [[ -z "$DATASET" && "$PRIMARY_CHECKPOINT" == *"/checkpoints/"* ]]; then
  DATASET="$(echo "$PRIMARY_CHECKPOINT" | awk -F'/checkpoints/' '{print $1}' | awk -F'/' '{print $NF}')"
fi
if [[ -z "$DATASET" ]]; then
  DATASET="$(
    config_fields --field dataset_name
  )"
fi
[[ -z "$DATASET" ]] && DATASET="DuEE-Fin"

mapfile -t _dataset_ctx < <(
  config_fields --field dataset_dir --field schema_path --field split_prefix
)
DATASET_DIR="${_dataset_ctx[0]}"
SCHEMA_FILE="${_dataset_ctx[1]}"
SPLIT_FILE="${DATASET_DIR}/${_dataset_ctx[2]}_${SPLIT}.json"
if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="logs/${DATASET}/eval_academic/${RUN_TS}_${SPLIT}"
  if [[ -n "$TAG" ]]; then
    OUT_DIR="${OUT_DIR}_${TAG}"
  fi
fi
mkdir -p "$OUT_DIR"

# Basic preflight checks (fail fast with actionable messages)
if [[ ! -f "evaluate.py" ]]; then
  echo "ERROR: evaluate.py not found in project root: $ROOT_DIR"
  exit 1
fi
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "ERROR: dataset dir missing: $DATASET_DIR"
  echo "Hint: sync script excludes data/ by default; copy dataset to server first."
  exit 1
fi
if [[ ! -f "$SPLIT_FILE" ]]; then
  echo "ERROR: split file missing: $SPLIT_FILE"
  exit 1
fi
if [[ ! -f "$SCHEMA_FILE" ]]; then
  echo "ERROR: schema file missing: $SCHEMA_FILE"
  exit 1
fi
if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import importlib
for m in ("torch", "yaml", "tqdm"):
    importlib.import_module(m)
PY
then
  echo "ERROR: python dependencies missing in current environment."
  echo "Hint: activate env and run 'pip install -r requirements.txt && pip install -e .'"
  exit 1
fi

echo "============================================================"
echo "OG-LANS"
echo "============================================================"
echo "checkpoint: $CHECKPOINT"
echo "dataset:    $DATASET"
echo "data_dir:   $DATASET_DIR"
echo "config:     $CONFIG"
echo "protocol:   $PROTOCOL"
echo "split:      $SPLIT"
echo "eval_mode:  $EVAL_MODE"
echo "batch_size: $BATCH_SIZE"
echo "seeds:      $SEEDS"
echo "seed_policy:$SEED_POLICY"
echo "checkpoints:${CHECKPOINT_ARR[*]}"
echo "num_samples:${NUM_SAMPLES:-ALL}"
echo "oneshot:    $USE_ONESHOT"
echo "prompt_variant:${PROMPT_VARIANT:-<from config>}"
echo "fewshot_num_examples:${FEWSHOT_NUM_EXAMPLES:-<from config>}"
echo "do_sample:  $DO_SAMPLE"
echo "cot_eval_mode: $COT_EVAL_MODE"
echo "pipeline_mode: $PIPELINE_MODE"
echo "output dir: $OUT_DIR"
echo "tail_on_fail: $TAIL_ON_FAIL lines"
echo "============================================================"

MANIFEST_JSON="${OUT_DIR}/run_manifest.json"
"$PYTHON_BIN" - "$MANIFEST_JSON" "$RUN_TS" "$CHECKPOINT" "$SEED_POLICY" "$DATASET" "$DATASET_DIR" "$SCHEMA_FILE" "$SPLIT_FILE" "$CONFIG" "$PROTOCOL" "$ROLE_ALIAS_MAP" "$CANONICAL_MODE" "$PRIMARY_METRIC" "$SPLIT" "$EVAL_MODE" "$BATCH_SIZE" "$SEEDS" "${NUM_SAMPLES:-ALL}" "$USE_ONESHOT" "$DO_SAMPLE" "$COT_EVAL_MODE" "$PIPELINE_MODE" "$OUT_DIR" "${PROMPT_VARIANT:-}" "${FEWSHOT_NUM_EXAMPLES:-}" "$ORIGINAL_CMD" <<'PY'
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys

(
    manifest_path,
    run_ts,
    checkpoint_csv,
    seed_policy,
    dataset_name,
    dataset_dir,
    schema_file,
    split_file,
    config_path,
    protocol_path,
    role_alias_map,
    canonical_mode,
    primary_metric,
    split,
    eval_mode,
    batch_size,
    seeds,
    num_samples,
    use_oneshot,
    do_sample,
    cot_eval_mode,
    pipeline_mode,
    out_dir,
    prompt_variant,
    fewshot_num_examples,
    command,
) = sys.argv[1:]

checkpoints = [x.strip() for x in checkpoint_csv.split(",") if x.strip()]
abs_checkpoints = [os.path.abspath(x) for x in checkpoints]

def safe_git(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None

config_hash = None
try:
    with open(config_path, "rb") as f:
        config_hash = hashlib.sha256(f.read()).hexdigest()
except Exception:
    pass

manifest = {
    "task": "eval_academic",
    "status": "running",
    "run_ts": run_ts,
    "command": command,
    "checkpoint": abs_checkpoints[0] if abs_checkpoints else None,
    "checkpoints": abs_checkpoints,
    "seed_policy": seed_policy,
    "dataset": dataset_name,
    "dataset_dir": os.path.abspath(dataset_dir),
    "schema_file": os.path.abspath(schema_file),
    "split_file": os.path.abspath(split_file),
    "config_path": os.path.abspath(config_path),
    "config_hash_sha256": config_hash,
    "protocol_path": os.path.abspath(protocol_path),
    "role_alias_map": os.path.abspath(role_alias_map) if role_alias_map else None,
    "canonical_metric_mode": canonical_mode,
    "primary_metric": primary_metric,
    "split": split,
    "eval_mode": eval_mode,
    "batch_size": int(batch_size),
    "seeds": [s.strip() for s in seeds.split(",") if s.strip()],
    "num_samples": num_samples,
    "use_oneshot": int(use_oneshot),
    "prompt_variant": prompt_variant or ("fewshot" if int(use_oneshot) else "zeroshot"),
    "fewshot_num_examples": int(fewshot_num_examples) if fewshot_num_examples else (3 if int(use_oneshot) else 0),
    "do_sample": int(do_sample),
    "cot_eval_mode": cot_eval_mode,
    "pipeline_mode": pipeline_mode,
    "artifacts": {
        "output_dir": os.path.abspath(out_dir),
    },
    "runtime_manifest": {
        "python": {
            "version": platform.python_version(),
            "executable": os.path.abspath(sys.executable),
        },
        "system": {
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
        },
        "git": {
            "commit": safe_git(["git", "rev-parse", "HEAD"]),
            "dirty": bool(safe_git(["git", "status", "--porcelain"]) or ""),
        },
    },
}

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
PY

declare -a SUCCESS_SEEDS
declare -a FAILED_SEEDS

for idx in "${!SEED_ARR[@]}"; do
  seed_raw="${SEED_ARR[$idx]}"
  seed="$(echo "$seed_raw" | xargs)"
  [[ -z "$seed" ]] && continue
  checkpoint_for_seed="$PRIMARY_CHECKPOINT"
  if [[ "$SEED_POLICY" == "train_seed" ]]; then
    checkpoint_for_seed="${CHECKPOINT_ARR[$idx]}"
  fi

  result_file="${OUT_DIR}/eval_results_${SPLIT}_seed${seed}.jsonl"
  metrics_file="${OUT_DIR}/eval_results_${SPLIT}_seed${seed}_metrics.json"
  log_file="${OUT_DIR}/run_seed${seed}.log"

  cmd=(
    "$PYTHON_BIN" evaluate.py
    --config "$CONFIG"
    --protocol "$PROTOCOL"
    --checkpoint "$checkpoint_for_seed"
    --seed "$seed"
    --split "$SPLIT"
    --batch_size "$BATCH_SIZE"
    --eval_mode "$EVAL_MODE"
    --role_alias_map "$ROLE_ALIAS_MAP"
    --canonical_metric_mode "$CANONICAL_MODE"
    --cot_eval_mode "$COT_EVAL_MODE"
    --pipeline_mode "$PIPELINE_MODE"
    --output_file "$result_file"
  )
  if [[ -n "$PRIMARY_METRIC" ]]; then
    cmd+=(--report_primary_metric "$PRIMARY_METRIC")
  fi
  if [[ -n "$NUM_SAMPLES" ]]; then
    cmd+=(--num_samples "$NUM_SAMPLES")
  fi
  if [[ "$USE_ONESHOT" -eq 1 ]]; then
    cmd+=(--use_oneshot)
  fi
  if [[ -n "$PROMPT_VARIANT" ]]; then
    cmd+=(--prompt_variant "$PROMPT_VARIANT")
  fi
  if [[ -n "$FEWSHOT_NUM_EXAMPLES" ]]; then
    cmd+=(--fewshot_num_examples "$FEWSHOT_NUM_EXAMPLES")
  fi
  if [[ "$DO_SAMPLE" -eq 1 ]]; then
    cmd+=(--do_sample)
  fi

  echo
  echo ">>> Seed ${seed}"
  echo "Checkpoint: ${checkpoint_for_seed}"
  echo "CMD: ${cmd[*]}"
  echo "LOG: ${log_file}"

  set +e
  "${cmd[@]}" >"$log_file" 2>&1
  rc=$?
  set -e

  if [[ $rc -ne 0 ]]; then
    echo "Seed ${seed} failed (exit=${rc})"
    if [[ -f "$log_file" ]]; then
      echo "----- Last ${TAIL_ON_FAIL} lines of ${log_file} -----"
      tail -n "$TAIL_ON_FAIL" "$log_file" || true
      echo "------------------------------------------------------"
    else
      echo "Log file not found: $log_file"
    fi
    FAILED_SEEDS+=("$seed")
    echo "Stopping due to failure."
    exit $rc
  fi

  if [[ ! -f "$metrics_file" ]]; then
    echo "Seed ${seed} finished but metrics file missing: $metrics_file"
    FAILED_SEEDS+=("$seed")
    exit 2
  fi

  SUCCESS_SEEDS+=("$seed")
done

echo
echo "Successful seeds: ${SUCCESS_SEEDS[*]:-none}"
echo "Failed seeds:     ${FAILED_SEEDS[*]:-none}"

if [[ ${#SUCCESS_SEEDS[@]} -eq 0 ]]; then
  "$PYTHON_BIN" - "$MANIFEST_JSON" "${FAILED_SEEDS[*]:-}" "$SEED_POLICY" <<'PY'
import json
import sys

manifest_path, failed_raw, seed_policy = sys.argv[1:]
failed_seeds = [x for x in failed_raw.split() if x]
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)
manifest["status"] = "failed"
manifest["seed_policy"] = seed_policy
manifest["results"] = {"successful_seeds": [], "failed_seeds": [int(x) for x in failed_seeds]}
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
PY
  echo "ERROR: no successful runs."
  exit 3
fi

"$PYTHON_BIN" - "$OUT_DIR" "$SEED_POLICY" "${SUCCESS_SEEDS[@]}" <<'PY'
import json
import math
import statistics
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
seed_policy = sys.argv[2]
seeds = sys.argv[3:]

def read_metrics(seed: str):
    p = out_dir / f"eval_results_dev_seed{seed}_metrics.json"
    if not p.exists():
        raise FileNotFoundError(f"metrics missing for seed={seed}: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f), p.name

rows = []
files = []
for s in seeds:
    m, fn = read_metrics(s)
    files.append(fn)
    rows.append({
        "seed": int(s),
        "doc_role_micro_f1": float(m.get("doc_role_micro_f1", m.get("academic_metrics", {}).get("doc_ee", {}).get("overall", {}).get("MicroF1", 0.0))),
        "doc_instance_micro_f1": float(m.get("doc_instance_micro_f1", m.get("academic_metrics", {}).get("doc_ee", {}).get("instance", {}).get("MicroF1", 0.0))),
        "doc_combination_micro_f1": float(m.get("doc_combination_micro_f1", m.get("academic_metrics", {}).get("doc_ee", {}).get("combination", {}).get("MicroF1", 0.0))),
        "doc_event_type_micro_f1": float(m.get("doc_event_type_micro_f1", m.get("academic_metrics", {}).get("doc_ee", {}).get("classification", {}).get("MicroF1", 0.0))),
        "strict_f1": float(m["strict"]["f1"]),
        "relaxed_f1": float(m["relaxed"]["f1"]),
        "type_f1": float(m["type_identification"]["f1"]),
        "parse_error_rate": float(m["parse_statistics"]["parse_error_rate"]),
        "hallucination_rate": float(m["hallucination"]["sample_rate"]),
        "schema_compliance_rate": float(m["schema_compliance_rate"]),
    })

def mean_std(vals):
    if len(vals) == 1:
        return vals[0], 0.0
    return statistics.mean(vals), statistics.stdev(vals)

keys = [
    "doc_role_micro_f1",
    "doc_instance_micro_f1",
    "doc_combination_micro_f1",
    "doc_event_type_micro_f1",
    "strict_f1",
    "relaxed_f1",
    "type_f1",
    "parse_error_rate",
    "hallucination_rate",
    "schema_compliance_rate",
]
agg = {}
for k in keys:
    vals = [r[k] for r in rows]
    mu, sd = mean_std(vals)
    agg[k] = {
        "mean": round(mu, 6),
        "std": round(sd, 6),
        "min": round(min(vals), 6),
        "max": round(max(vals), 6),
    }

summary = {
    "n_runs": len(rows),
    "seed_policy": seed_policy,
    "seeds": [r["seed"] for r in rows],
    "metrics_files": files,
    "aggregate": agg,
    "per_seed": rows,
}

out_json = out_dir / "academic_summary.json"
out_md = out_dir / "academic_summary.md"

with out_json.open("w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

lines = []
lines.append("# OG-LANS 评估汇总")
lines.append("")
lines.append(f"- Runs: {summary['n_runs']}")
lines.append(f"- Seed policy: {summary['seed_policy']}")
lines.append(f"- Seeds: {summary['seeds']}")
lines.append("")
lines.append("| Metric | Mean | Std | Min | Max |")
lines.append("|---|---:|---:|---:|---:|")
for k in keys:
    a = agg[k]
    lines.append(f"| {k} | {a['mean']:.4f} | {a['std']:.4f} | {a['min']:.4f} | {a['max']:.4f} |")
lines.append("")
lines.append("## Per-seed")
lines.append("")
lines.append("| Seed | doc_role_micro_f1 | doc_instance_micro_f1 | doc_combination_micro_f1 | doc_event_type_micro_f1 | strict_f1 | relaxed_f1 | type_f1 | parse_error_rate | hallucination_rate | schema_compliance_rate |")
lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in sorted(rows, key=lambda x: x["seed"]):
    lines.append(
        f"| {r['seed']} | {r['doc_role_micro_f1']:.4f} | {r['doc_instance_micro_f1']:.4f} | "
        f"{r['doc_combination_micro_f1']:.4f} | {r['doc_event_type_micro_f1']:.4f} | "
        f"{r['strict_f1']:.4f} | {r['relaxed_f1']:.4f} | {r['type_f1']:.4f} | "
        f"{r['parse_error_rate']:.4f} | {r['hallucination_rate']:.4f} | {r['schema_compliance_rate']:.4f} |"
    )

with out_md.open("w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print(f"Saved: {out_json}")
print(f"Saved: {out_md}")
PY

"$PYTHON_BIN" - "$MANIFEST_JSON" "$OUT_DIR" "${SUCCESS_SEEDS[*]:-}" "${FAILED_SEEDS[*]:-}" "$SEED_POLICY" <<'PY'
import json
import os
import sys

manifest_path, out_dir, success_raw, failed_raw, seed_policy = sys.argv[1:]
success_seeds = [x for x in success_raw.split() if x]
failed_seeds = [x for x in failed_raw.split() if x]

with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

manifest["status"] = "completed"
manifest["seed_policy"] = seed_policy
manifest["results"] = {
    "successful_seeds": [int(x) for x in success_seeds],
    "failed_seeds": [int(x) for x in failed_seeds],
}
manifest.setdefault("artifacts", {})
manifest["artifacts"].update({
    "output_dir": os.path.abspath(out_dir),
    "summary_json": os.path.abspath(os.path.join(out_dir, "academic_summary.json")),
    "summary_md": os.path.abspath(os.path.join(out_dir, "academic_summary.md")),
})

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
PY

echo
echo "Done."
echo "Output directory: $OUT_DIR"
echo "Files:"
echo "  - run_manifest.json"
echo "  - run_seed<seed>.log"
echo "  - eval_results_<split>_seed<seed>.jsonl"
echo "  - eval_results_<split>_seed<seed>_metrics.json"
echo "  - academic_summary.json"
echo "  - academic_summary.md"
