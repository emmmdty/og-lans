#!/usr/bin/env bash
set -euo pipefail

# scripts/run_train.sh
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
PROJECT_ROOT="$ROOT_DIR"
# ==========================================
DATASET_NAME="DuEE-Fin"
# ==========================================

# Runtime environment
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="offline"
export PYTHONUNBUFFERED="1"  # 确保 Python 输出不缓存，实时写入日志
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-${PROJECT_ROOT}/models}"
usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_train.sh [options] [-- extra main.py args]

Core options:
  --config <path>         Config path. Default: configs/config.yaml
  --data_dir <path>       Dataset directory. Default: ./data/raw/DuEE-Fin
  --schema_path <path>    Optional schema override
  --exp_name <name>       Optional experiment name
  -h, --help              Show help

All other options are forwarded to main.py unchanged.
EOF
}
resolve_python_bin() {
  if command -v uv >/dev/null 2>&1; then
    echo "uv run python"
    return
  fi
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
resolve_python_cmd() {
  local resolved
  resolved="$(resolve_python_bin)"
  if [[ -z "$resolved" ]]; then
    return
  fi
  if [[ "$resolved" == "uv run python" ]]; then
    PYTHON_CMD=(uv run python)
    PYTHON_DISPLAY="uv run python"
    return
  fi
  PYTHON_CMD=("$resolved")
  PYTHON_DISPLAY="$resolved"
}
resolve_gpu_banner() {
  local gpu_info=""
  if [[ ${#PYTHON_CMD[@]} -gt 0 ]]; then
    gpu_info="$("${PYTHON_CMD[@]}" - <<'PY' 2>/dev/null
import json
try:
    import torch
except Exception:
    print(json.dumps({"gpu_summary": "GPU unknown", "cuda": "unknown", "torch": "unknown"}))
    raise SystemExit(0)

summary = []
if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        total_gb = props.total_memory / (1024 ** 3)
        summary.append(f"GPU{idx}:{torch.cuda.get_device_name(idx)} ({total_gb:.1f}GB)")

payload = {
    "gpu_summary": " | ".join(summary) if summary else "GPU unknown",
    "cuda": str(getattr(torch.version, "cuda", None) or "unknown"),
    "torch": str(getattr(torch, "__version__", "unknown")),
}
print(json.dumps(payload, ensure_ascii=False))
PY
)"
  fi

  if [[ -n "$gpu_info" ]]; then
    local parsed
    parsed="$("${PYTHON_CMD[@]}" - <<'PY' "$gpu_info"
import json
import sys
payload = json.loads(sys.argv[1])
print(payload["gpu_summary"])
print(payload["cuda"])
print(payload["torch"])
PY
)"
    if [[ -n "$parsed" ]]; then
      printf '%s\n' "$parsed"
      return
    fi
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -n 1
    echo "unknown"
    "${PYTHON_CMD[@]}" - <<'PY' 2>/dev/null || echo "unknown"
try:
    import torch
    print(torch.__version__)
except Exception:
    raise SystemExit(1)
PY
    return
  fi

  printf 'GPU unknown\nunknown\nunknown\n'
}
PYTHON_CMD=()
PYTHON_DISPLAY=""
resolve_python_cmd
if [[ ${#PYTHON_CMD[@]} -eq 0 ]]; then
  echo "ERROR: unable to resolve Python runtime (expected uv run python, python, or python3)."
  exit 1
fi
# 手动创建目录，防止报错
mkdir -p "$MODELSCOPE_CACHE"

echo "📦 Model Root Dir: $MODELSCOPE_CACHE"

CONFIG_PATH="${PROJECT_ROOT}/configs/config.yaml"
# 默认数据目录
DATA_DIR="${PROJECT_ROOT}/data/raw/${DATASET_NAME}"

# === 解析命令行参数，用于脚本内部的日志同步 ===
USER_EXP_NAME=""
USER_HAS_DATA_DIR=0
USER_HAS_CONFIG=0
USER_SCHEMA_PATH=""
ORIGINAL_CMD="bash $0 $*"
args=("$@")
for ((i=0; i<${#args[@]}; i++)); do
  if [ "${args[$i]}" == "-h" ] || [ "${args[$i]}" == "--help" ]; then
    usage
    exit 0
  fi
  if [ "${args[$i]}" == "--data_dir" ] || [ "${args[$i]}" == "--data-dir" ]; then
    DATA_DIR="${args[$((i+1))]}"
    USER_HAS_DATA_DIR=1
  elif [ "${args[$i]}" == "--config" ]; then
    CONFIG_PATH="${args[$((i+1))]}"
    USER_HAS_CONFIG=1
  elif [ "${args[$i]}" == "--schema_path" ] || [ "${args[$i]}" == "--schema-path" ]; then
    USER_SCHEMA_PATH="${args[$((i+1))]}"
  elif [ "${args[$i]}" == "--exp_name" ] || [ "${args[$i]}" == "--exp-name" ]; then
    USER_EXP_NAME="${args[$((i+1))]}"
  fi
done

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: config not found: $CONFIG_PATH"
  exit 1
fi
if [[ -n "$USER_SCHEMA_PATH" && ! -f "$USER_SCHEMA_PATH" ]]; then
  echo "ERROR: schema file not found: $USER_SCHEMA_PATH"
  exit 1
fi

# 更新显示用的数据集名称
DATASET_NAME=$(basename "$DATA_DIR")
mapfile -t GPU_BANNER_LINES < <(resolve_gpu_banner)
GPU_SUMMARY="${GPU_BANNER_LINES[0]:-GPU unknown}"
CUDA_VERSION="${GPU_BANNER_LINES[1]:-unknown}"
TORCH_VERSION="${GPU_BANNER_LINES[2]:-unknown}"

echo "=========================================================="
echo "   OG-LANS                                                  "
echo "   Environment: ${GPU_SUMMARY} | CUDA ${CUDA_VERSION} | torch ${TORCH_VERSION}"
echo "   Python=${PYTHON_DISPLAY}"
echo "   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "   MODELSCOPE_CACHE=${MODELSCOPE_CACHE}"
echo "=========================================================="

CHECKPOINT_ROOT="${PROJECT_ROOT}/logs/${DATASET_NAME}/checkpoints"
mkdir -p "$CHECKPOINT_ROOT"

if [ -n "$USER_EXP_NAME" ]; then
    EXP_NAME="$USER_EXP_NAME"
else
    counter=1
    while [ -d "${CHECKPOINT_ROOT}/exp${counter}" ]; do
      counter=$((counter + 1))
    done
    EXP_NAME="exp${counter}"
fi

# === Auto-Log + Run Manifest ===
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="${TIMESTAMP}_${EXP_NAME}"
RUN_DIR="${PROJECT_ROOT}/logs/${DATASET_NAME}/train/${RUN_ID}"
mkdir -p "$RUN_DIR"

LOG_FILE="${RUN_DIR}/run.log"
RUN_MANIFEST="${RUN_DIR}/run_manifest.json"
echo "📝 Terminal output will be saved to: $LOG_FILE"
echo "🧾 Run manifest: $RUN_MANIFEST"

"${PYTHON_CMD[@]}" - "$RUN_MANIFEST" "$TIMESTAMP" "$ORIGINAL_CMD" "$CONFIG_PATH" "$DATA_DIR" "$EXP_NAME" "$RUN_DIR" "$USER_SCHEMA_PATH" -- "$@" <<'PY'
import json
import os
import sys

from oglans.utils.run_manifest import collect_runtime_manifest, load_effective_config_metadata

argv = sys.argv[1:]
manifest_path, ts, cmd, config_path, data_dir, exp_name, run_dir, schema_path = argv[:8]
forwarded_args = argv[9:] if len(argv) > 8 and argv[8] == "--" else []

config_hash = None
seed = None
training_mode = None
lans_enabled = None
scv_enabled = None
stage_mode = None
fewshot_selection_mode = None
fewshot_pool_split = None
train_tune_ratio = None
research_split_manifest_path = None
teacher_silver_enabled = None
teacher_silver_path = None
teacher_silver_max_samples = None
try:
    config_meta = load_effective_config_metadata(config_path, cli_args=forwarded_args)
    config_hash = config_meta["config_hash_sha256"]
    seed = config_meta["seed"]
    training_mode = config_meta["training_mode"]
    lans_enabled = config_meta["lans_enabled"]
    scv_enabled = config_meta["scv_enabled"]
    stage_mode = config_meta["stage_mode"]
    fewshot_selection_mode = config_meta["fewshot_selection_mode"]
    fewshot_pool_split = config_meta["fewshot_pool_split"]
    train_tune_ratio = config_meta["train_tune_ratio"]
    research_split_manifest_path = config_meta["research_split_manifest_path"]
    teacher_silver_enabled = config_meta["teacher_silver_enabled"]
    teacher_silver_path = config_meta["teacher_silver_path"]
    teacher_silver_max_samples = config_meta["teacher_silver_max_samples"]
except Exception:
    pass

manifest = {
    "task": "train",
    "status": "running",
    "run_id": os.path.basename(run_dir),
    "timestamp": ts,
    "command": cmd,
    "config_path": os.path.abspath(config_path),
    "config_hash_sha256": config_hash,
    "data_dir": os.path.abspath(data_dir),
    "schema_path": os.path.abspath(schema_path) if schema_path else None,
    "experiment": exp_name,
    "seed": seed,
    "training_mode": training_mode,
    "lans_enabled": lans_enabled,
    "scv_enabled": scv_enabled,
    "stage_mode": stage_mode,
    "fewshot_selection_mode": fewshot_selection_mode,
    "fewshot_pool_split": fewshot_pool_split,
    "train_tune_ratio": train_tune_ratio,
    "research_split_manifest_path": research_split_manifest_path,
    "teacher_silver_enabled": teacher_silver_enabled,
    "teacher_silver_path": teacher_silver_path,
    "teacher_silver_max_samples": teacher_silver_max_samples,
    "artifacts": {
        "run_dir": os.path.abspath(run_dir),
        "log_file": os.path.abspath(os.path.join(run_dir, "run.log")),
    },
    "runtime_manifest": collect_runtime_manifest(
        os.getcwd(),
        package_names=["torch", "transformers", "trl", "unsloth", "datasets", "PyYAML"],
    ),
}

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
PY

# 重定向 stdout 和 stderr 到文件，同时保留终端显示
exec > >(tee -a "$LOG_FILE") 2>&1
# ==============================

"${PYTHON_CMD[@]}" - "$CONFIG_PATH" -- "$@" <<'PY'
import sys

from oglans.utils.run_manifest import load_effective_config_metadata

argv = sys.argv[1:]
config_path = argv[0]
forwarded_args = argv[2:] if len(argv) > 1 and argv[1] == "--" else []
try:
    cfg = load_effective_config_metadata(config_path, cli_args=forwarded_args)["config"]
except Exception as exc:
    print(f"[WARN] Failed to parse config summary: {exc}")
    sys.exit(0)

t = cfg.get("training", {})
a = cfg.get("algorithms", {})
lans = a.get("lans", {})
scv = a.get("scv", {})
rpo = t.get("rpo", {})
pref = t.get("preference", {})
st = lans.get("strategies", {})

print("🧪 Config Summary:")
print(f"  seed={cfg.get('project', {}).get('seed')}")
print(f"  max_steps={t.get('max_steps', -1)} | epochs={t.get('num_train_epochs')}")
print(f"  logging_steps={t.get('logging_steps')}")
print(f"  LANS={lans.get('enabled')} | SCV={scv.get('enabled')}")
print(
    "  RPO alpha="
    f"{rpo.get('alpha', 0.0)} | warmup_steps={rpo.get('warmup_steps', 0)}"
)
print(
    "  Preference mode="
    f"{pref.get('mode', 'ipo')} | offset_source={pref.get('offset_source', 'margin_bucket')} "
    f"| offset_static={pref.get('offset_static', 0.15)}"
)
print(
    "  LANS refresh_start_epoch="
    f"{lans.get('refresh_start_epoch', 1)} | refresh_log_interval={lans.get('refresh_log_interval', 200)}"
)
print(
    "  LANS hard_floor="
    f"{st.get('hard_floor_prob', 0.0)} -> {st.get('hard_floor_after_warmup', st.get('hard_floor_prob', 0.0))}"
)
print(f"  Stage mode={cfg.get('comparison', {}).get('stage_mode', 'single_pass')}")
print(
    "  Few-shot selection="
    f"{cfg.get('comparison', {}).get('fewshot_selection_mode', 'dynamic')} "
    f"| pool_split={cfg.get('comparison', {}).get('fewshot_pool_split', 'train_fit')}"
)
print(
    "  Research split manifest="
    f"{cfg.get('comparison', {}).get('research_split_manifest_path')}"
)
print(
    "  Teacher silver="
    f"{cfg.get('training', {}).get('teacher_silver', {}).get('enabled', False)} "
    f"| path={cfg.get('training', {}).get('teacher_silver', {}).get('path')} "
    f"| max_samples={cfg.get('training', {}).get('teacher_silver', {}).get('max_samples')}"
)
PY

echo "=========================================================="
echo "   🚀 Dataset: ${DATASET_NAME}"
echo "   🧪 Experiment: ${EXP_NAME}"
if [[ -n "$USER_SCHEMA_PATH" ]]; then
echo "   🗂️ Schema: ${USER_SCHEMA_PATH}"
fi
echo "   📂 Log Path: logs/${DATASET_NAME}/checkpoints/${EXP_NAME}"
echo "=========================================================="

# 运行主程序
# 参数规则：
# 1) 用户显式传入的参数原样透传；
# 2) 若用户未传 --data_dir，则补默认 DATA_DIR；
# 3) 若用户未传 --exp_name，则补自动生成的 EXP_NAME。
RUN_START_EPOCH="$(date +%s)"
set +e
MAIN_CMD=("${PYTHON_CMD[@]}" main.py)
if [[ "$USER_HAS_CONFIG" -eq 0 ]]; then
  MAIN_CMD+=(--config "$CONFIG_PATH")
fi
if [[ "$USER_HAS_DATA_DIR" -eq 0 ]]; then
  MAIN_CMD+=(--data_dir "$DATA_DIR")
fi
if [[ -z "$USER_EXP_NAME" ]]; then
  MAIN_CMD+=(--exp_name "$EXP_NAME")
fi
MAIN_CMD+=("$@")
"${MAIN_CMD[@]}"
RC=$?
set -e
RUN_END_EPOCH="$(date +%s)"

"${PYTHON_CMD[@]}" - "$RUN_MANIFEST" "$RC" "$RUN_START_EPOCH" "$RUN_END_EPOCH" "$CHECKPOINT_ROOT" "$EXP_NAME" <<'PY'
import json
import os
import sys

manifest_path, rc, start_epoch, end_epoch, checkpoint_root, exp_name = sys.argv[1:]
with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

exit_code = int(rc)
status = "completed" if exit_code == 0 else "failed"
manifest["status"] = status
manifest["runtime"] = {
    "start_epoch": int(start_epoch),
    "end_epoch": int(end_epoch),
    "wall_clock_seconds": int(end_epoch) - int(start_epoch),
    "exit_code": exit_code,
}
manifest.setdefault("artifacts", {})
manifest["artifacts"]["checkpoint_dir"] = os.path.abspath(os.path.join(checkpoint_root, exp_name))

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
PY

if [[ $RC -ne 0 ]]; then
  echo "[ERROR] Experiment ${EXP_NAME} failed (exit=${RC})."
  exit $RC
fi

echo "[SUCCESS] Experiment ${EXP_NAME} completed."
