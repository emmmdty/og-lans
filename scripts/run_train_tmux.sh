#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

SESSION="oglans_train"
CONDA_ENV="tjk_ee"
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
CONFIG="configs/config.yaml"
DATA_DIR="./data/raw/DuEE-Fin"
EXP_NAME=""
SCHEMA_PATH=""
ATTACH=0
FORCE_RESTART=0
DRY_RUN=0
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Run OG-LANS training inside tmux.

Usage:
  bash scripts/run_train_tmux.sh [options] [-- extra run_train args]

Options:
  --session <name>      tmux session name (default: oglans_train)
  --env <name>          conda env name (default: tjk_ee)
  --gpu <id>            CUDA_VISIBLE_DEVICES value (default: env or 0)
  --config <path>       training config (default: configs/config.yaml)
  --data_dir <path>     dataset dir (default: ./data/raw/DuEE-Fin)
  --exp_name <name>     experiment name (optional)
  --schema_path <path>  schema path override (optional)
  --attach              attach tmux after launch
  --force-restart       kill existing same-name session before start
  --dry-run             print launch script and exit
  -h, --help            show help

Example:
  bash scripts/run_train_tmux.sh \
    --session train_main_s3407 \
    --env tjk_ee \
    --gpu 0 \
    --config configs/config.yaml \
    --data_dir ./data/raw/DuEE-Fin \
    --exp_name main_s3407 \
    --attach
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session) SESSION="${2:-}"; shift 2 ;;
    --env) CONDA_ENV="${2:-}"; shift 2 ;;
    --gpu) GPU_ID="${2:-}"; shift 2 ;;
    --config) CONFIG="${2:-}"; shift 2 ;;
    --data_dir|--data-dir) DATA_DIR="${2:-}"; shift 2 ;;
    --exp_name|--exp-name) EXP_NAME="${2:-}"; shift 2 ;;
    --schema_path|--schema-path) SCHEMA_PATH="${2:-}"; shift 2 ;;
    --attach) ATTACH=1; shift ;;
    --force-restart) FORCE_RESTART=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux not found."
  exit 1
fi
if [[ -z "$SESSION" ]]; then
  echo "ERROR: --session cannot be empty."
  exit 1
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "ERROR: config not found: $CONFIG"
  exit 1
fi
if [[ ! -d "$DATA_DIR" ]]; then
  echo "ERROR: data_dir not found: $DATA_DIR"
  exit 1
fi
if [[ -n "$SCHEMA_PATH" && ! -f "$SCHEMA_PATH" ]]; then
  echo "ERROR: schema_path not found: $SCHEMA_PATH"
  exit 1
fi

TMUX_LOG_DIR="${ROOT_DIR}/logs/tmux"
mkdir -p "$TMUX_LOG_DIR"
LOG_FILE="${TMUX_LOG_DIR}/${SESSION}_$(date +%Y%m%d_%H%M%S).log"
LAUNCHER="${TMUX_LOG_DIR}/${SESSION}_launch.sh"

train_cmd=(bash scripts/run_train.sh --config "$CONFIG" --data_dir "$DATA_DIR")
if [[ -n "$EXP_NAME" ]]; then
  train_cmd+=(--exp_name "$EXP_NAME")
fi
if [[ -n "$SCHEMA_PATH" ]]; then
  train_cmd+=(--schema_path "$SCHEMA_PATH")
fi
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  train_cmd+=("${EXTRA_ARGS[@]}")
fi

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  echo "cd $(printf '%q' "$ROOT_DIR")"
  echo "source ~/.bashrc"
  echo "conda activate $(printf '%q' "$CONDA_ENV")"
  echo "export CUDA_VISIBLE_DEVICES=$(printf '%q' "$GPU_ID")"
  echo "export PYTHONUNBUFFERED=1"
  echo 'echo "[tmux-train] start: $(date "+%F %T")"'
  echo "cmd=("
  for arg in "${train_cmd[@]}"; do
    printf "  %q\n" "$arg"
  done
  echo ")"
  printf '"${cmd[@]}" 2>&1 | tee -a %q\n' "$LOG_FILE"
  echo 'rc=${PIPESTATUS[0]}'
  echo 'echo "[tmux-train] exit_code=${rc} end: $(date "+%F %T")"'
  echo "exit \$rc"
} > "$LAUNCHER"

chmod +x "$LAUNCHER"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  if [[ "$FORCE_RESTART" -eq 1 ]]; then
    tmux kill-session -t "$SESSION"
  else
    echo "ERROR: tmux session already exists: $SESSION"
    echo "Use --force-restart or choose another --session name."
    exit 1
  fi
fi

echo "Session:   $SESSION"
echo "Conda env: $CONDA_ENV"
echo "GPU:       $GPU_ID"
echo "Log file:  $LOG_FILE"
echo "Launcher:  $LAUNCHER"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo
  echo "--- launcher ---"
  cat "$LAUNCHER"
  exit 0
fi

tmux new-session -d -s "$SESSION" "bash $(printf '%q' "$LAUNCHER")"
echo "Started tmux session: $SESSION"
echo "Attach: tmux attach -t $SESSION"
echo "Tail log: tail -f $(printf '%q' "$LOG_FILE")"

if [[ "$ATTACH" -eq 1 ]]; then
  tmux attach -t "$SESSION"
fi
