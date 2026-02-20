#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
PROJECT_ROOT="$ROOT_DIR"

# Runtime environment
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="offline"
export PYTHONUNBUFFERED="1"  # Ensure realtime logging without Python stdout buffering
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

# =========================
# Defaults
# =========================
ACTION="run"                    # run | preflight | sweep
CONFIG="configs/config.yaml"
PROTOCOL="configs/eval_protocol.yaml"
SPLIT="dev"
MODEL="deepseek-chat"
CONCURRENCY="8"
NUM_SAMPLES=""
SEED=""
FEWSHOT="0"                     # 0 | 1
JSON_MODE="auto"                # auto | on | off
COMPUTE_CI="1"                  # 1 | 0
ROLE_ALIAS_MAP="configs/role_aliases_duee_fin.yaml"
CANONICAL_METRIC_MODE="analysis_only"  # off | analysis_only | apply_for_aux_metric
PRIMARY_METRIC=""               # empty => protocol default
OUTPUT_FILE=""
SUMMARY_FILE=""
BACKGROUND="0"                  # 0 | 1

# Sweep-only
SEEDS="3407,3408,3409"
VARIANTS="zeroshot,fewshot"     # comma-separated

timestamp() {
  date +%Y%m%d_%H%M%S
}

infer_dataset_context() {
  "$PYTHON_BIN" - "$CONFIG" <<'PY'
import os
import yaml
import sys

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

ds_cns = cfg.get("algorithms", {}).get("ds_cns", {})
taxonomy = ds_cns.get("taxonomy_path", "")

dataset_dir = ""
if taxonomy:
    dataset_dir = os.path.dirname(os.path.normpath(str(taxonomy)))
else:
    project = cfg.get("project", {})
    cache_dir = project.get("dataset_cache_dir", "")
    if cache_dir:
        cache_norm = os.path.normpath(str(cache_dir))
        base = os.path.basename(cache_norm)
        if base and base.lower() not in {"cache", "processed", "data"}:
            dataset_dir = os.path.normpath(os.path.join("data", "raw", base))
if not dataset_dir:
    dataset_dir = os.path.normpath(os.path.join("data", "raw", "DuEE-Fin"))

dataset_name = os.path.basename(os.path.normpath(dataset_dir)) or "DuEE-Fin"
split_prefix = "duee_fin"

def infer_eval_root(config, default_dataset):
    project = config.get("project", {})
    for key in ("output_dir", "logging_dir", "debug_data_dir"):
        raw = project.get(key)
        if not raw:
            continue
        norm = os.path.normpath(str(raw)).replace("\\", "/")
        parts = [p for p in norm.split("/") if p and p != "."]
        if "logs" not in parts:
            continue
        idx = parts.index("logs")
        if idx + 1 < len(parts):
            tag = parts[idx + 1]
            if tag:
                return os.path.normpath(os.path.join("logs", tag, "eval_api"))
    return os.path.normpath(os.path.join("logs", default_dataset, "eval_api"))

eval_root = infer_eval_root(cfg, dataset_name)

print(dataset_dir)
print(dataset_name)
print(taxonomy)
print(split_prefix)
print(eval_root)
PY
}

usage() {
  cat <<'EOF'
Unified evaluate_api launcher.

Usage:
  bash scripts/run_eval_api.sh [options]

Default behavior:
  action=run, split=dev, model=deepseek-chat, concurrency=8, zeroshot

Actions:
  -a, --action <run|preflight|sweep>
      run       Run one evaluation job (default).
      preflight Check dataset/key/dependencies only.
      sweep     Run multiple jobs by seeds/variants on one split.

Core options:
  -c, --config <path>          Config path. Default: configs/config.yaml
      --protocol <path>        Eval protocol path. Default: configs/eval_protocol.yaml
  -s, --split <dev|test|train> Dataset split. Default: dev
  -m, --model <name>           Model name. Default: deepseek-chat
  -j, --concurrency <int>      API concurrency. Default: 8
  -n, --num-samples <int>      Evaluate first N samples
      --seed <int>             Random seed (run action)
  -f, --fewshot                Enable few-shot
  -z, --zeroshot               Force zero-shot
      --json-mode <auto|on|off>
      --role-alias-map <path>  Role alias map path
      --canonical-mode <off|analysis_only|apply_for_aux_metric>
      --primary-metric <name>  Primary metric key for reporting
      --no-ci                  Disable bootstrap CI
      --output-file <path>     Output JSONL path
      --summary-file <path>    Summary JSON path
  -b, --background             Run in background with nohup (run action only)

Sweep options:
      --seeds <csv>            e.g. 3407,3408,3409
      --variants <csv>         zeroshot,fewshot (default both)

Convenience:
      --smoke                  Shortcut for run with --num-samples 20 --concurrency 2
  -h, --help                   Show this help

Examples:
  bash scripts/run_eval_api.sh --action preflight
  bash scripts/run_eval_api.sh
  bash scripts/run_eval_api.sh -f --split dev
  bash scripts/run_eval_api.sh --smoke
  bash scripts/run_eval_api.sh --action sweep --split dev --seeds 3407,3408 --variants zeroshot,fewshot
EOF
}

join_cmd() {
  local out=""
  local part
  for part in "$@"; do
    if [[ -z "$out" ]]; then
      out="$part"
    else
      out="$out $part"
    fi
  done
  echo "$out"
}

preflight() {
  echo "[1/4] Check files..."
  test -f "evaluate_api.py"
  test -f "$CONFIG"
  test -f "$PROTOCOL"
  if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import yaml  # noqa: F401
PY
  then
    echo "ERROR: missing dependency 'yaml' in current Python environment."
    exit 1
  fi

  local ds_dir ds_name schema_path split_prefix split_file eval_root
  mapfile -t ctx < <(infer_dataset_context)
  ds_dir="${ctx[0]}"
  ds_name="${ctx[1]}"
  schema_path="${ctx[2]}"
  split_prefix="${ctx[3]}"
  eval_root="${ctx[4]}"
  split_file="${ds_dir}/${split_prefix}_${SPLIT}.json"

  echo "[2/4] Check dataset..."
  if [[ ! -d "$ds_dir" ]]; then
    echo "ERROR: missing dataset directory: $ds_dir"
    exit 1
  fi
  if [[ -n "$schema_path" && ! -f "$schema_path" ]]; then
    echo "ERROR: missing schema file from config taxonomy_path: $schema_path"
    exit 1
  fi
  if [[ ! -f "$split_file" ]]; then
    echo "ERROR: missing split file: $split_file"
    exit 1
  fi
  echo "Dataset context: name=${ds_name}, dir=${ds_dir}, eval_root=${eval_root}"

  echo "[3/4] Check API key (env or .env)..."
  if [[ -n "${DEEPSEEK_API_KEY:-}" || -n "${OPENAI_API_KEY:-}" ]]; then
    echo "API key found in process environment."
  else
    if [[ ! -f ".env" ]]; then
      echo "ERROR: neither env var nor .env exists."
      exit 1
    fi
    if grep -Eq '^(DEEPSEEK_API_KEY|OPENAI_API_KEY)\s*=' .env; then
      echo "API key entry found in .env."
    else
      echo "ERROR: .env exists but no DEEPSEEK_API_KEY/OPENAI_API_KEY entry found."
      exit 1
    fi
  fi

  echo "[4/4] Check Python deps..."
  "$PYTHON_BIN" - <<'PY'
import importlib
for m in ["yaml", "openai", "tqdm"]:
    importlib.import_module(m)
print("Dependency check passed.")
PY

  echo "Preflight OK."
}

build_run_cmd() {
  local cmd=("$PYTHON_BIN" evaluate_api.py
    --config "$CONFIG"
    --protocol "$PROTOCOL"
    --split "$SPLIT"
    --model "$MODEL"
    --concurrency "$CONCURRENCY"
    --json_mode "$JSON_MODE"
    --role_alias_map "$ROLE_ALIAS_MAP"
    --canonical_metric_mode "$CANONICAL_METRIC_MODE"
  )

  if [[ -n "$NUM_SAMPLES" ]]; then
    cmd+=(--num_samples "$NUM_SAMPLES")
  fi
  if [[ -n "$SEED" ]]; then
    cmd+=(--seed "$SEED")
  fi
  if [[ "$FEWSHOT" == "1" ]]; then
    cmd+=(--use_fewshot)
  fi
  if [[ "$COMPUTE_CI" == "0" ]]; then
    cmd+=(--no-compute_ci)
  fi
  if [[ -n "$PRIMARY_METRIC" ]]; then
    cmd+=(--report_primary_metric "$PRIMARY_METRIC")
  fi
  if [[ -n "$OUTPUT_FILE" ]]; then
    cmd+=(--output_file "$OUTPUT_FILE")
  fi
  if [[ -n "$SUMMARY_FILE" ]]; then
    cmd+=(--summary_file "$SUMMARY_FILE")
  fi

  echo "$(join_cmd "${cmd[@]}")"
}

run_once() {
  local cmd
  local eval_root
  cmd="$(build_run_cmd)"
  mapfile -t ctx < <(infer_dataset_context)
  eval_root="${ctx[4]}"

  if [[ "$BACKGROUND" == "1" ]]; then
    mkdir -p "$eval_root"
    local log_path="${eval_root}/nohup_eval_api_$(timestamp).log"
    echo "Running in background:"
    echo "  $cmd"
    nohup bash -lc "$cmd" > "$log_path" 2>&1 &
    echo "Started PID: $!"
    echo "Log file: $log_path"
  else
    echo "Running:"
    echo "  $cmd"
    bash -lc "$cmd"
  fi
}

sweep() {
  if [[ "$BACKGROUND" == "1" ]]; then
    echo "ERROR: --background is not supported for --action sweep."
    exit 1
  fi
  if [[ -n "$OUTPUT_FILE" || -n "$SUMMARY_FILE" ]]; then
    echo "WARN: --output-file/--summary-file are ignored in sweep mode."
  fi

  IFS=',' read -r -a seed_arr <<< "$SEEDS"
  IFS=',' read -r -a var_arr <<< "$VARIANTS"

  local eval_root
  mapfile -t ctx < <(infer_dataset_context)
  eval_root="${ctx[4]}"
  local sweep_root="${eval_root}/sweep_$(timestamp)"
  mkdir -p "$sweep_root"

  local v s ts out sum old_few old_seed old_out old_sum
  for v in "${var_arr[@]}"; do
    v="$(echo "$v" | xargs)"
    if [[ "$v" != "zeroshot" && "$v" != "fewshot" ]]; then
      echo "ERROR: invalid variant in --variants: $v (allowed: zeroshot,fewshot)"
      exit 1
    fi
    for s in "${seed_arr[@]}"; do
      s="$(echo "$s" | xargs)"
      if [[ -z "$s" ]]; then
        continue
      fi

      ts="$(timestamp)"
      out="${sweep_root}/eval_results_deepseek_${SPLIT}_${v}_seed${s}_${ts}.jsonl"
      sum="${sweep_root}/eval_summary_${SPLIT}_${v}_seed${s}_${ts}.json"

      old_few="$FEWSHOT"
      old_seed="$SEED"
      old_out="$OUTPUT_FILE"
      old_sum="$SUMMARY_FILE"

      if [[ "$v" == "fewshot" ]]; then
        FEWSHOT="1"
      else
        FEWSHOT="0"
      fi
      SEED="$s"
      OUTPUT_FILE="$out"
      SUMMARY_FILE="$sum"

      echo "========================================"
      echo "Sweep job: split=$SPLIT variant=$v seed=$s"
      echo "========================================"
      run_once

      FEWSHOT="$old_few"
      SEED="$old_seed"
      OUTPUT_FILE="$old_out"
      SUMMARY_FILE="$old_sum"
    done
  done
}

if [[ $# -eq 0 ]]; then
  :
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    -a|--action)
      ACTION="${2:-}"; shift 2 ;;
    -c|--config)
      CONFIG="${2:-}"; shift 2 ;;
    --protocol)
      PROTOCOL="${2:-}"; shift 2 ;;
    -s|--split)
      SPLIT="${2:-}"; shift 2 ;;
    -m|--model)
      MODEL="${2:-}"; shift 2 ;;
    -j|--concurrency)
      CONCURRENCY="${2:-}"; shift 2 ;;
    -n|--num-samples)
      NUM_SAMPLES="${2:-}"; shift 2 ;;
    --seed)
      SEED="${2:-}"; shift 2 ;;
    -f|--fewshot)
      FEWSHOT="1"; shift ;;
    -z|--zeroshot)
      FEWSHOT="0"; shift ;;
    --json-mode)
      JSON_MODE="${2:-}"; shift 2 ;;
    --role-alias-map)
      ROLE_ALIAS_MAP="${2:-}"; shift 2 ;;
    --canonical-mode)
      CANONICAL_METRIC_MODE="${2:-}"; shift 2 ;;
    --primary-metric)
      PRIMARY_METRIC="${2:-}"; shift 2 ;;
    --no-ci)
      COMPUTE_CI="0"; shift ;;
    --output-file)
      OUTPUT_FILE="${2:-}"; shift 2 ;;
    --summary-file)
      SUMMARY_FILE="${2:-}"; shift 2 ;;
    -b|--background)
      BACKGROUND="1"; shift ;;
    --seeds)
      SEEDS="${2:-}"; shift 2 ;;
    --variants)
      VARIANTS="${2:-}"; shift 2 ;;
    --smoke)
      ACTION="run"
      NUM_SAMPLES="20"
      CONCURRENCY="2"
      shift ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "Unknown option: $1"
      echo
      usage
      exit 1 ;;
  esac
done

case "$ACTION" in
  preflight)
    preflight ;;
  run)
    run_once ;;
  sweep)
    sweep ;;
  *)
    echo "ERROR: unknown --action: $ACTION"
    usage
    exit 1 ;;
esac
