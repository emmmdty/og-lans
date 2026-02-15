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
export HF_HOME="${PROJECT_ROOT}/data/cache/huggingface"
export HF_DATASETS_CACHE="${PROJECT_ROOT}/data/cache/huggingface/datasets"
# 手动创建目录，防止报错
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

echo "📦 HuggingFace Cache Dir: $HF_DATASETS_CACHE"

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

echo "=========================================================="
echo "   OG-LANS                                                  "
echo "   Environment: A6000 (48GB) | CUDA 11.8 | torch 2.6.0    "
echo "   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
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

python - "$RUN_MANIFEST" "$TIMESTAMP" "$ORIGINAL_CMD" "$CONFIG_PATH" "$DATA_DIR" "$EXP_NAME" "$RUN_DIR" "$USER_SCHEMA_PATH" <<'PY'
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys

manifest_path, ts, cmd, config_path, data_dir, exp_name, run_dir, schema_path = sys.argv[1:]

def safe_git(cmdline):
    try:
        return subprocess.check_output(cmdline, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None

config_hash = None
seed = None
try:
    import yaml  # type: ignore
    with open(config_path, "rb") as f:
        raw = f.read()
        config_hash = hashlib.sha256(raw).hexdigest()
    cfg = yaml.safe_load(raw.decode("utf-8"))
    seed = cfg.get("project", {}).get("seed")
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
    "artifacts": {
        "run_dir": os.path.abspath(run_dir),
        "log_file": os.path.abspath(os.path.join(run_dir, "run.log")),
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

# 重定向 stdout 和 stderr 到文件，同时保留终端显示
exec > >(tee -a "$LOG_FILE") 2>&1
# ==============================

python - "$CONFIG_PATH" <<'PY'
import sys

config_path = sys.argv[1]
try:
    import yaml  # type: ignore
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
except Exception as exc:
    print(f"[WARN] Failed to parse config summary: {exc}")
    sys.exit(0)

t = cfg.get("training", {})
a = cfg.get("algorithms", {})
lans = a.get("lans", {})
scv = a.get("scv", {})

print("🧪 Config Summary:")
print(f"  seed={cfg.get('project', {}).get('seed')}")
print(f"  max_steps={t.get('max_steps', -1)} | epochs={t.get('num_train_epochs')}")
print(f"  logging_steps={t.get('logging_steps')}")
print(f"  LANS={lans.get('enabled')} | SCV={scv.get('enabled')}")
print(
    "  LANS refresh_start_epoch="
    f"{lans.get('refresh_start_epoch', 1)} | refresh_log_interval={lans.get('refresh_log_interval', 200)}"
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
MAIN_CMD=(python main.py)
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

python - "$RUN_MANIFEST" "$RC" "$RUN_START_EPOCH" "$RUN_END_EPOCH" "$CHECKPOINT_ROOT" "$EXP_NAME" <<'PY'
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
