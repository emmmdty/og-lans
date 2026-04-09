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
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${PROJECT_ROOT}/data/cache/modelscope}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/data/cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-${HF_HOME}/assets}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"

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
mkdir -p "$MODELSCOPE_CACHE" "$HF_HOME" "$HF_HUB_CACHE" "$HF_ASSETS_CACHE" "$HF_XET_CACHE"

config_fields() {
  "$PYTHON_BIN" scripts/resolve_config_context.py --config "$CONFIG" --project-root "$PROJECT_ROOT" "$@"
}

CONFIG="configs/config.yaml"
PROTOCOL="configs/eval_protocol.yaml"
ROLE_ALIAS_MAP="configs/role_aliases_duee_fin.yaml"
CANONICAL_MODE="analysis_only"
PRIMARY_METRIC=""
MODEL_NAME=""
DATASET_NAME=""
EXP_NAME=""
SPLIT="dev"
NUM_SAMPLES=""
BATCH_SIZE="16"
PROMPT_VARIANT=""
FEWSHOT_NUM_EXAMPLES=""
STAGE_MODE=""
FEWSHOT_SELECTION_MODE=""
FEWSHOT_POOL_SPLIT=""
TRAIN_TUNE_RATIO=""
RESEARCH_SPLIT_MANIFEST=""
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_eval_base.sh [options]

Options:
  --model-name <path>      Optional. Override base model path/name (e.g., Qwen/...).
  --dataset-name <name>    Optional. Dataset name used for output directory.
  --config <path>          Config path. Default: configs/config.yaml
  --protocol <path>        Eval protocol path. Default: configs/eval_protocol.yaml
  --role-alias-map <path>  Role alias map. Default: configs/role_aliases_duee_fin.yaml
  --canonical-mode <mode>  off|analysis_only|apply_for_aux_metric
  --primary-metric <name>  Primary metric key (optional, overrides protocol)
  --exp-name <name>        Experiment name used for log/result filenames.
  --split <name>           Dataset split. Default: dev
  --num-samples <int>      Optional. Limit evaluation to the first N samples.
  --batch-size <int>       Inference batch size. Default: 16
  --prompt-variant <mode>  zeroshot|fewshot (optional)
  --fewshot-num-examples <int>
                           Few-shot example count (optional)
  --stage-mode <mode>      single_pass|two_stage (optional)
  --fewshot-selection-mode <mode>
                           static|dynamic (optional)
  --fewshot-pool-split <mode>
                           train|train_fit (optional)
  --train-tune-ratio <float>
                           Optional. train_tune ratio used when no frozen manifest is provided.
  --research-split-manifest <path>
                           Optional. Frozen train_fit/train_tune split manifest path.
  -h, --help               Show help.

Other evaluate.py args are forwarded transparently.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-name|--model_name|--model_name_or_path)
      MODEL_NAME="${2:-}"; shift 2 ;;
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
    --num-samples|--num_samples)
      NUM_SAMPLES="${2:-}"; shift 2 ;;
    --batch-size|--batch_size)
      BATCH_SIZE="${2:-}"; shift 2 ;;
    --prompt-variant|--prompt_variant)
      PROMPT_VARIANT="${2:-}"; shift 2 ;;
    --fewshot-num-examples|--fewshot_num_examples)
      FEWSHOT_NUM_EXAMPLES="${2:-}"; shift 2 ;;
    --stage-mode|--stage_mode)
      STAGE_MODE="${2:-}"; shift 2 ;;
    --fewshot-selection-mode|--fewshot_selection_mode)
      FEWSHOT_SELECTION_MODE="${2:-}"; shift 2 ;;
    --fewshot-pool-split|--fewshot_pool_split)
      FEWSHOT_POOL_SPLIT="${2:-}"; shift 2 ;;
    --train-tune-ratio|--train_tune_ratio)
      TRAIN_TUNE_RATIO="${2:-}"; shift 2 ;;
    --research-split-manifest|--research_split_manifest)
      RESEARCH_SPLIT_MANIFEST="${2:-}"; shift 2 ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      EXTRA_ARGS+=("$1")
      shift ;;
  esac
done

if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG"
  exit 1
fi
if [[ ! -f "$PROTOCOL" ]]; then
  echo "ERROR: protocol not found: $PROTOCOL"
  exit 1
fi

if [[ -z "$DATASET_NAME" ]]; then
  DATASET_NAME="$(
    config_fields --field dataset_name
  )"
fi
[[ -z "$DATASET_NAME" ]] && DATASET_NAME="DuEE-Fin"

if [[ -z "$EXP_NAME" ]]; then
  if [[ -n "$MODEL_NAME" ]]; then
    EXP_NAME="$(basename "$MODEL_NAME")"
  else
    EXP_NAME="base_model"
  fi
  EXP_NAME="${EXP_NAME//\//_}"
  EXP_NAME="${EXP_NAME//:/_}"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${TIMESTAMP}_${EXP_NAME}_${SPLIT}"
RUN_DIR="${PROJECT_ROOT}/logs/${DATASET_NAME}/eval_base/${RUN_ID}"
mkdir -p "$RUN_DIR"

LOG_FILE="${RUN_DIR}/run.log"
OUTPUT_JSONL="${RUN_DIR}/eval_results.jsonl"

echo "Evaluation log: $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================================="
echo "OG-LANS Base Model Evaluation"
echo "time:       $TIMESTAMP"
echo "config:     $CONFIG"
echo "protocol:   $PROTOCOL"
echo "model_name: ${MODEL_NAME:-<from config>}"
echo "split:      $SPLIT"
echo "num_samples:${NUM_SAMPLES:-<all>}"
echo "batch_size: $BATCH_SIZE"
echo "output:     $OUTPUT_JSONL"
echo "run_dir:    $RUN_DIR"
echo "=========================================================="

cmd=(
  "$PYTHON_BIN" evaluate.py
  --base_only
  --config "$CONFIG"
  --protocol "$PROTOCOL"
  --split "$SPLIT"
  --batch_size "$BATCH_SIZE"
  --role_alias_map "$ROLE_ALIAS_MAP"
  --canonical_metric_mode "$CANONICAL_MODE"
  --output_file "$OUTPUT_JSONL"
)
if [[ -n "$NUM_SAMPLES" ]]; then
  cmd+=(--num_samples "$NUM_SAMPLES")
fi
if [[ -n "$MODEL_NAME" ]]; then
  cmd+=(--model_name_or_path "$MODEL_NAME")
fi
if [[ -n "$PRIMARY_METRIC" ]]; then
  cmd+=(--report_primary_metric "$PRIMARY_METRIC")
fi
if [[ -n "$PROMPT_VARIANT" ]]; then
  cmd+=(--prompt_variant "$PROMPT_VARIANT")
fi
if [[ -n "$FEWSHOT_NUM_EXAMPLES" ]]; then
  cmd+=(--fewshot_num_examples "$FEWSHOT_NUM_EXAMPLES")
fi
if [[ -n "$STAGE_MODE" ]]; then
  cmd+=(--stage_mode "$STAGE_MODE")
fi
if [[ -n "$FEWSHOT_SELECTION_MODE" ]]; then
  cmd+=(--fewshot_selection_mode "$FEWSHOT_SELECTION_MODE")
fi
if [[ -n "$FEWSHOT_POOL_SPLIT" ]]; then
  cmd+=(--fewshot_pool_split "$FEWSHOT_POOL_SPLIT")
fi
if [[ -n "$TRAIN_TUNE_RATIO" ]]; then
  cmd+=(--train_tune_ratio "$TRAIN_TUNE_RATIO")
fi
if [[ -n "$RESEARCH_SPLIT_MANIFEST" ]]; then
  cmd+=(--research_split_manifest "$RESEARCH_SPLIT_MANIFEST")
fi
cmd+=("${EXTRA_ARGS[@]}")
"${cmd[@]}"

echo "=========================================================="
echo "Evaluation completed."
echo "log:    $LOG_FILE"
echo "result: $OUTPUT_JSONL"
echo "=========================================================="
