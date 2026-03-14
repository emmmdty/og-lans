#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
PROJECT_ROOT="$ROOT_DIR"

export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE="offline"
export PYTHONUNBUFFERED="1"

EVAL_GPU="0"
TRAIN_GPU="1"
EVAL_BATCH_SIZE="8"
EVAL_SPLIT="dev"
BASE_MODEL_NAME=""
DATA_DIR="./data/raw/DuEE-Fin"
EXP_NAME=""
TRAIN_PER_DEVICE_BS="2"
TRAIN_GRAD_ACCUM="16"
TRAIN_EVAL_ENABLED="1"
TRAIN_EVAL_SPLIT="dev"
TRAIN_EVAL_BATCH_SIZE="16"
TRAIN_EVAL_CHECKPOINT=""
TRAIN_EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_parallel_eval_train.sh [options] [-- <extra train args>]

Runs in parallel:
  - GPU0 (default): full dev base-model evaluation via run_eval_base.sh
  - GPU1 (default): fine-tuning training via run_train.sh, then local eval on same GPU

Options:
  --eval-gpu <id>               Eval GPU id. Default: 0
  --train-gpu <id>              Train GPU id. Default: 1
  --eval-batch-size <int>       Base eval batch size. Default: 8
  --eval-split <name>           Eval split. Default: dev
  --base-model-name <path/name> Optional base model override for run_eval_base.sh
  --data-dir <path>             Training data dir. Default: ./data/raw/DuEE-Fin
  --exp-name <name>             Training experiment name. Default: parallel_<timestamp>
  --train-batch-size <int>      training.per_device_train_batch_size. Default: 2
  --grad-accum <int>            training.gradient_accumulation_steps. Default: 16
  --no-train-eval               Disable automatic post-train local evaluation on GPU1
  --train-eval-split <name>     Post-train local eval split. Default: dev
  --train-eval-batch-size <int> Post-train local eval batch size. Default: 16
  --train-eval-checkpoint <p>   Override checkpoint path for post-train local eval
  -h, --help                    Show help

Examples:
  bash scripts/run_parallel_eval_train.sh --exp-name main_s3407_v4
  bash scripts/run_parallel_eval_train.sh --eval-batch-size 16 --exp-name exp1
  bash scripts/run_parallel_eval_train.sh --exp-name exp2 -- --training.num_train_epochs 3
  bash scripts/run_parallel_eval_train.sh --exp-name exp3 --train-eval-batch-size 8
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval-gpu)
      EVAL_GPU="${2:-}"; shift 2 ;;
    --train-gpu)
      TRAIN_GPU="${2:-}"; shift 2 ;;
    --eval-batch-size|--eval_batch_size)
      EVAL_BATCH_SIZE="${2:-}"; shift 2 ;;
    --eval-split|--split)
      EVAL_SPLIT="${2:-}"; shift 2 ;;
    --base-model-name|--base_model_name|--model-name)
      BASE_MODEL_NAME="${2:-}"; shift 2 ;;
    --data-dir|--data_dir)
      DATA_DIR="${2:-}"; shift 2 ;;
    --exp-name|--exp_name)
      EXP_NAME="${2:-}"; shift 2 ;;
    --train-batch-size)
      TRAIN_PER_DEVICE_BS="${2:-}"; shift 2 ;;
    --grad-accum|--gradient-accumulation-steps)
      TRAIN_GRAD_ACCUM="${2:-}"; shift 2 ;;
    --no-train-eval)
      TRAIN_EVAL_ENABLED="0"; shift ;;
    --train-eval-split)
      TRAIN_EVAL_SPLIT="${2:-}"; shift 2 ;;
    --train-eval-batch-size|--train_eval_batch_size)
      TRAIN_EVAL_BATCH_SIZE="${2:-}"; shift 2 ;;
    --train-eval-checkpoint|--train_eval_checkpoint)
      TRAIN_EVAL_CHECKPOINT="${2:-}"; shift 2 ;;
    --)
      shift
      TRAIN_EXTRA_ARGS+=("$@")
      break ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "ERROR: Unknown argument: $1"
      usage
      exit 1 ;;
  esac
done

if [[ -z "$EXP_NAME" ]]; then
  EXP_NAME="parallel_$(date +%m%d_%H%M%S)"
fi

if [[ "$EVAL_GPU" == "$TRAIN_GPU" ]]; then
  echo "ERROR: --eval-gpu and --train-gpu must be different."
  exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: data dir not found: $DATA_DIR"
  exit 1
fi

TRAIN_DATASET_NAME="$(basename "$(realpath "$DATA_DIR")")"
if [[ -z "$TRAIN_EVAL_CHECKPOINT" ]]; then
  TRAIN_EVAL_CHECKPOINT="logs/${TRAIN_DATASET_NAME}/checkpoints/${EXP_NAME}"
fi

TS="$(date +%Y%m%d_%H%M%S)"
OPS_DIR="${PROJECT_ROOT}/logs/ops/${TS}_evalgpu${EVAL_GPU}_traingpu${TRAIN_GPU}_${EXP_NAME}"
mkdir -p "$OPS_DIR"

EVAL_LOG="${OPS_DIR}/eval_base_gpu${EVAL_GPU}.log"
TRAIN_LOG="${OPS_DIR}/train_gpu${TRAIN_GPU}.log"
TRAIN_EVAL_LOG="${OPS_DIR}/eval_local_after_train_gpu${TRAIN_GPU}.log"
GPU1_STATUS_FILE="${OPS_DIR}/gpu1_pipeline_status.txt"
META_FILE="${OPS_DIR}/run_meta.txt"

{
  echo "time=$TS"
  echo "project_root=$PROJECT_ROOT"
  echo "eval_gpu=$EVAL_GPU"
  echo "train_gpu=$TRAIN_GPU"
  echo "eval_batch_size=$EVAL_BATCH_SIZE"
  echo "eval_split=$EVAL_SPLIT"
  echo "base_model_name=${BASE_MODEL_NAME:-<from config>}"
  echo "data_dir=$DATA_DIR"
  echo "exp_name=$EXP_NAME"
  echo "train_per_device_bs=$TRAIN_PER_DEVICE_BS"
  echo "train_grad_accum=$TRAIN_GRAD_ACCUM"
  echo "train_eval_enabled=$TRAIN_EVAL_ENABLED"
  echo "train_eval_split=$TRAIN_EVAL_SPLIT"
  echo "train_eval_batch_size=$TRAIN_EVAL_BATCH_SIZE"
  echo "train_eval_checkpoint=$TRAIN_EVAL_CHECKPOINT"
  echo "train_extra_args=${TRAIN_EXTRA_ARGS[*]:-}"
} > "$META_FILE"

echo "=========================================================="
echo "Parallel run launcher"
echo "  eval  GPU : $EVAL_GPU (base model, split=$EVAL_SPLIT, bs=$EVAL_BATCH_SIZE)"
echo "  train GPU : $TRAIN_GPU (exp=$EXP_NAME)"
if [[ "$TRAIN_EVAL_ENABLED" == "1" ]]; then
  echo "  train->eval: enabled (split=$TRAIN_EVAL_SPLIT, bs=$TRAIN_EVAL_BATCH_SIZE)"
  echo "  train eval checkpoint: $TRAIN_EVAL_CHECKPOINT"
else
  echo "  train->eval: disabled"
fi
echo "  ops dir   : $OPS_DIR"
echo "  eval log  : $EVAL_LOG"
echo "  train log : $TRAIN_LOG"
if [[ "$TRAIN_EVAL_ENABLED" == "1" ]]; then
  echo "  train eval log: $TRAIN_EVAL_LOG"
fi
echo "=========================================================="

cleanup_children() {
  local rc=$?
  trap - INT TERM EXIT
  if [[ -n "${EVAL_PID:-}" ]]; then kill "$EVAL_PID" 2>/dev/null || true; fi
  if [[ -n "${GPU1_PIPE_PID:-}" ]]; then kill "$GPU1_PIPE_PID" 2>/dev/null || true; fi
  exit "$rc"
}
trap cleanup_children INT TERM

eval_cmd=(
  bash scripts/run_eval_base.sh
  --split "$EVAL_SPLIT"
  --batch-size "$EVAL_BATCH_SIZE"
)
if [[ -n "$BASE_MODEL_NAME" ]]; then
  eval_cmd+=(--model-name "$BASE_MODEL_NAME")
fi

train_cmd=(
  bash scripts/run_train.sh
  --data_dir "$DATA_DIR"
  --exp_name "$EXP_NAME"
  --training.per_device_train_batch_size "$TRAIN_PER_DEVICE_BS"
  --training.gradient_accumulation_steps "$TRAIN_GRAD_ACCUM"
)
if [[ ${#TRAIN_EXTRA_ARGS[@]} -gt 0 ]]; then
  train_cmd+=("${TRAIN_EXTRA_ARGS[@]}")
fi

train_eval_cmd=(
  bash scripts/run_eval_local.sh
  --checkpoint "$TRAIN_EVAL_CHECKPOINT"
  --split "$TRAIN_EVAL_SPLIT"
  --batch-size "$TRAIN_EVAL_BATCH_SIZE"
)

echo "[START] eval base model on GPU${EVAL_GPU}"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" "${eval_cmd[@]}" >"$EVAL_LOG" 2>&1 &
EVAL_PID=$!
echo "  PID=$EVAL_PID"

sleep 3

echo "[START] train pipeline on GPU${TRAIN_GPU} (train -> local eval)"
(
  set +e
  TRAIN_STAGE_RC=0
  TRAIN_EVAL_STAGE_RC=0

  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "${train_cmd[@]}" >"$TRAIN_LOG" 2>&1
  TRAIN_STAGE_RC=$?

  if [[ "$TRAIN_STAGE_RC" -eq 0 && "$TRAIN_EVAL_ENABLED" == "1" ]]; then
    CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "${train_eval_cmd[@]}" >"$TRAIN_EVAL_LOG" 2>&1
    TRAIN_EVAL_STAGE_RC=$?
  elif [[ "$TRAIN_EVAL_ENABLED" == "1" ]]; then
    TRAIN_EVAL_STAGE_RC=99
  fi

  {
    echo "train_stage_rc=$TRAIN_STAGE_RC"
    echo "train_eval_stage_rc=$TRAIN_EVAL_STAGE_RC"
  } > "$GPU1_STATUS_FILE"

  if [[ "$TRAIN_STAGE_RC" -ne 0 ]]; then
    exit "$TRAIN_STAGE_RC"
  fi
  if [[ "$TRAIN_EVAL_ENABLED" == "1" && "$TRAIN_EVAL_STAGE_RC" -ne 0 ]]; then
    exit "$TRAIN_EVAL_STAGE_RC"
  fi
  exit 0
) &
GPU1_PIPE_PID=$!
echo "  PID=$GPU1_PIPE_PID"

echo
echo "Monitor:"
echo "  tail -f \"$EVAL_LOG\""
echo "  tail -f \"$TRAIN_LOG\""
if [[ "$TRAIN_EVAL_ENABLED" == "1" ]]; then
  echo "  tail -f \"$TRAIN_EVAL_LOG\""
fi
echo "  watch -n 5 nvidia-smi"
echo

set +e
wait "$EVAL_PID"; EVAL_RC=$?
wait "$GPU1_PIPE_PID"; GPU1_PIPE_RC=$?
set -e

TRAIN_RC="NA"
TRAIN_EVAL_RC="NA"
if [[ -f "$GPU1_STATUS_FILE" ]]; then
  TRAIN_RC="$(grep -E '^train_stage_rc=' "$GPU1_STATUS_FILE" | tail -1 | cut -d= -f2 || echo NA)"
  TRAIN_EVAL_RC="$(grep -E '^train_eval_stage_rc=' "$GPU1_STATUS_FILE" | tail -1 | cut -d= -f2 || echo NA)"
fi

{
  echo "eval_pid=$EVAL_PID"
  echo "gpu1_pipeline_pid=$GPU1_PIPE_PID"
  echo "eval_exit_code=$EVAL_RC"
  echo "gpu1_pipeline_exit_code=$GPU1_PIPE_RC"
  echo "train_stage_exit_code=$TRAIN_RC"
  echo "train_eval_stage_exit_code=$TRAIN_EVAL_RC"
} >> "$META_FILE"

echo "=========================================================="
echo "Finished:"
echo "  eval  exit code = $EVAL_RC"
echo "  gpu1 pipeline exit code = $GPU1_PIPE_RC"
echo "    - train stage exit code     = $TRAIN_RC"
if [[ "$TRAIN_EVAL_ENABLED" == "1" ]]; then
  echo "    - train eval stage exit code = $TRAIN_EVAL_RC"
fi
echo "=========================================================="

if [[ "$EVAL_RC" -ne 0 || "$GPU1_PIPE_RC" -ne 0 ]]; then
  exit 1
fi
