#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
PROJECT_ROOT="$ROOT_DIR"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="offline"
export PYTHONUNBUFFERED="1"
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

CONFIG="configs/config.yaml"
PROTOCOL="configs/eval_protocol.yaml"
ROLE_ALIAS_MAP="configs/role_aliases_duee_fin.yaml"
CANONICAL_MODE="analysis_only"
PRIMARY_METRIC=""
CHECKPOINT=""
DATASET_NAME=""
EXP_NAME=""
SPLIT="dev"
BATCH_SIZE="16"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_eval_local.sh --checkpoint <path> [options]

Options:
  --checkpoint <path>     Required. Checkpoint path.
  --dataset-name <name>   Optional. Dataset name used for output directory.
  --config <path>         Config path. Default: configs/config.yaml
  --protocol <path>       Eval protocol path. Default: configs/eval_protocol.yaml
  --role-alias-map <path> Role alias map. Default: configs/role_aliases_duee_fin.yaml
  --canonical-mode <mode> off|analysis_only|apply_for_aux_metric
  --primary-metric <name> Primary metric key (optional, overrides protocol)
  --exp-name <name>       Experiment name used for log/result filenames.
  --split <name>          Dataset split. Default: dev
  --batch-size <int>      Inference batch size. Default: 16
  -h, --help              Show help.

Other evaluate.py args are forwarded transparently.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="${2:-}"; shift 2 ;;
    --config)
      CONFIG="${2:-}"; shift 2 ;;
    --dataset-name)
      DATASET_NAME="${2:-}"; shift 2 ;;
    --protocol)
      PROTOCOL="${2:-}"; shift 2 ;;
    --role-alias-map)
      ROLE_ALIAS_MAP="${2:-}"; shift 2 ;;
    --canonical-mode)
      CANONICAL_MODE="${2:-}"; shift 2 ;;
    --primary-metric)
      PRIMARY_METRIC="${2:-}"; shift 2 ;;
    --exp-name|--exp_name)
      EXP_NAME="${2:-}"; shift 2 ;;
    --split)
      SPLIT="${2:-}"; shift 2 ;;
    --batch-size|--batch_size)
      BATCH_SIZE="${2:-}"; shift 2 ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      EXTRA_ARGS+=("$1")
      shift ;;
  esac
done

if [[ -z "$CHECKPOINT" ]]; then
  echo "ERROR: --checkpoint is required."
  usage
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG"
  exit 1
fi
if [[ ! -f "$PROTOCOL" ]]; then
  echo "ERROR: protocol not found: $PROTOCOL"
  exit 1
fi

if [[ -z "$DATASET_NAME" ]]; then
  if [[ "$CHECKPOINT" == *"/checkpoints/"* ]]; then
    DATASET_NAME="$(echo "$CHECKPOINT" | awk -F'/checkpoints/' '{print $1}' | awk -F'/' '{print $NF}')"
  fi
fi
if [[ -z "$DATASET_NAME" ]]; then
  DATASET_NAME="$(
    "$PYTHON_BIN" - "$CONFIG" <<'PY'
import os
import sys
import yaml

config_path = sys.argv[1]
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}
taxonomy = cfg.get("algorithms", {}).get("ds_cns", {}).get("taxonomy_path")
if taxonomy:
    dataset = os.path.basename(os.path.dirname(os.path.normpath(str(taxonomy))))
    print(dataset or "DuEE-Fin")
else:
    print("DuEE-Fin")
PY
  )"
fi
[[ -z "$DATASET_NAME" ]] && DATASET_NAME="DuEE-Fin"

if [[ -z "$EXP_NAME" ]]; then
  EXP_NAME="$(basename "$CHECKPOINT")"
  if [[ "$EXP_NAME" == checkpoint-* ]]; then
    EXP_NAME="$(basename "$(dirname "$CHECKPOINT")")"
  fi
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${TIMESTAMP}_${EXP_NAME}_${SPLIT}"
RUN_DIR="${PROJECT_ROOT}/logs/${DATASET_NAME}/eval_local/${RUN_ID}"
mkdir -p "$RUN_DIR"

LOG_FILE="${RUN_DIR}/run.log"
OUTPUT_JSONL="${RUN_DIR}/eval_results.jsonl"

echo "Evaluation log: $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================================="
echo "OG-LANS"
echo "time:       $TIMESTAMP"
echo "config:     $CONFIG"
echo "protocol:   $PROTOCOL"
echo "checkpoint: $CHECKPOINT"
echo "split:      $SPLIT"
echo "batch_size: $BATCH_SIZE"
echo "output:     $OUTPUT_JSONL"
echo "run_dir:    $RUN_DIR"
echo "=========================================================="

cmd=(
  "$PYTHON_BIN" evaluate.py
  --config "$CONFIG"
  --protocol "$PROTOCOL"
  --checkpoint "$CHECKPOINT"
  --split "$SPLIT"
  --batch_size "$BATCH_SIZE"
  --role_alias_map "$ROLE_ALIAS_MAP"
  --canonical_metric_mode "$CANONICAL_MODE"
  --output_file "$OUTPUT_JSONL"
)
if [[ -n "$PRIMARY_METRIC" ]]; then
  cmd+=(--report_primary_metric "$PRIMARY_METRIC")
fi
cmd+=("${EXTRA_ARGS[@]}")
"${cmd[@]}"

echo "=========================================================="
echo "Evaluation completed."
echo "log:    $LOG_FILE"
echo "result: $OUTPUT_JSONL"
echo "=========================================================="
