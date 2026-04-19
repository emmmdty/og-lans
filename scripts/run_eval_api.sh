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
export MODELSCOPE_CACHE="${PROJECT_ROOT}/models"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
resolve_python_bin() {
  if [[ -n "${CONDA_PREFIX:-}" ]] && command -v python >/dev/null 2>&1; then
    echo "python"
    return
  fi
  if [[ -n "${VIRTUAL_ENV:-}" ]] && command -v python >/dev/null 2>&1; then
    echo "python"
    return
  fi
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
resolve_python_cmd
if [[ -z "${PYTHON_DISPLAY:-}" ]]; then
  echo "ERROR: unable to resolve Python runtime (expected uv run python, python, or python3)."
  exit 1
fi
mkdir -p "$MODELSCOPE_CACHE"

# =========================
# Defaults
# =========================
ACTION="run"                    # run | preflight | sweep
CONFIG="configs/config.yaml"
PROTOCOL="configs/eval_protocol.yaml"
SPLIT="dev"
MODEL=""
BASE_URL=""
CONCURRENCY="8"
NUM_SAMPLES=""
SEED=""
FEWSHOT="0"                     # 0 | 1
STAGE_MODE="single_pass"        # single_pass | two_stage
FEWSHOT_SELECTION_MODE="dynamic" # static | dynamic
FEWSHOT_POOL_SPLIT="train_fit"  # train | train_fit
TRAIN_TUNE_RATIO=""             # optional, default from config
RESEARCH_SPLIT_MANIFEST=""      # optional frozen split manifest path
JSON_MODE="auto"                # auto | on | off
COT_EVAL_MODE="self_consistency" # self_consistency | counterfactual
PIPELINE_MODE="e2e"             # e2e | cat_lite | record_corrector | record_corrector+cat_lite
POSTPROCESS_PROFILE="none"      # none | event_probe_v2
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
  "${PYTHON_CMD[@]}" scripts/resolve_config_context.py --config "$CONFIG" --project-root "$PROJECT_ROOT" \
    --field dataset_dir \
    --field dataset_name \
    --field schema_path \
    --field split_prefix \
    --field eval_api_root
}

usage() {
  cat <<'EOF'
Unified evaluate_api launcher.

Usage:
  bash scripts/run_eval_api.sh [options]

Default behavior:
  action=run, split=dev, model=auto(cli/env/config), concurrency=8, zeroshot

Actions:
  -a, --action <run|preflight|sweep>
      run       Run one evaluation job (default).
      preflight Check dataset/key/dependencies only.
      sweep     Run multiple jobs by seeds/variants on one split.

Core options:
  -c, --config <path>          Config path. Default: configs/config.yaml
      --protocol <path>        Eval protocol path. Default: configs/eval_protocol.yaml
  -s, --split <dev|test|train> Dataset split. Default: dev
  -m, --model <name>           Model name. Default: auto(cli/env/config)
      --base-url <url>         Override API base URL
  -j, --concurrency <int>      API concurrency. Default: 8
  -n, --num-samples <int>      Evaluate first N samples
      --seed <int>             Random seed (run action)
  -f, --fewshot                Enable few-shot
  -z, --zeroshot               Force zero-shot
      --stage-mode <single_pass|two_stage|two_stage_per_type>
      --fewshot-selection-mode <static|dynamic>
      --fewshot-pool-split <train|train_fit>
      --train-tune-ratio <float>
      --research-split-manifest <path>
      --json-mode <auto|on|off>
      --cot-eval-mode <self_consistency|counterfactual>
      --pipeline-mode <e2e|cat_lite|record_corrector|record_corrector+cat_lite>
      --postprocess-profile <none|event_probe_v2>
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
  if ! "${PYTHON_CMD[@]}" - <<'PY' >/dev/null 2>&1
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
  "${PYTHON_CMD[@]}" - <<'PY'
import importlib
for m in ["yaml", "openai", "tqdm"]:
    importlib.import_module(m)
print("Dependency check passed.")
PY

  echo "Preflight OK."
}

build_run_cmd() {
  RUN_CMD=("${PYTHON_CMD[@]}" evaluate_api.py
    --config "$CONFIG"
    --protocol "$PROTOCOL"
    --split "$SPLIT"
    --concurrency "$CONCURRENCY"
    --json_mode "$JSON_MODE"
    --cot_eval_mode "$COT_EVAL_MODE"
    --pipeline_mode "$PIPELINE_MODE"
    --postprocess_profile "$POSTPROCESS_PROFILE"
    --role_alias_map "$ROLE_ALIAS_MAP"
    --canonical_metric_mode "$CANONICAL_METRIC_MODE"
  )

  if [[ -n "$MODEL" ]]; then
    RUN_CMD+=(--model "$MODEL")
  fi
  if [[ -n "$NUM_SAMPLES" ]]; then
    RUN_CMD+=(--num_samples "$NUM_SAMPLES")
  fi
  if [[ -n "$BASE_URL" ]]; then
    RUN_CMD+=(--base_url "$BASE_URL")
  fi
  if [[ -n "$SEED" ]]; then
    RUN_CMD+=(--seed "$SEED")
  fi
  if [[ "$FEWSHOT" == "1" ]]; then
    RUN_CMD+=(--use_fewshot)
  fi
  if [[ -n "$STAGE_MODE" ]]; then
    RUN_CMD+=(--stage_mode "$STAGE_MODE")
  fi
  if [[ -n "$FEWSHOT_SELECTION_MODE" ]]; then
    RUN_CMD+=(--fewshot_selection_mode "$FEWSHOT_SELECTION_MODE")
  fi
  if [[ -n "$FEWSHOT_POOL_SPLIT" ]]; then
    RUN_CMD+=(--fewshot_pool_split "$FEWSHOT_POOL_SPLIT")
  fi
  if [[ -n "$TRAIN_TUNE_RATIO" ]]; then
    RUN_CMD+=(--train_tune_ratio "$TRAIN_TUNE_RATIO")
  fi
  if [[ -n "$RESEARCH_SPLIT_MANIFEST" ]]; then
    RUN_CMD+=(--research_split_manifest "$RESEARCH_SPLIT_MANIFEST")
  fi
  if [[ "$COMPUTE_CI" == "0" ]]; then
    RUN_CMD+=(--no-compute_ci)
  fi
  if [[ -n "$PRIMARY_METRIC" ]]; then
    RUN_CMD+=(--report_primary_metric "$PRIMARY_METRIC")
  fi
  if [[ -n "$OUTPUT_FILE" ]]; then
    RUN_CMD+=(--output_file "$OUTPUT_FILE")
  fi
  if [[ -n "$SUMMARY_FILE" ]]; then
    RUN_CMD+=(--summary_file "$SUMMARY_FILE")
  fi
}

run_once() {
  local eval_root
  build_run_cmd
  mapfile -t ctx < <(infer_dataset_context)
  eval_root="${ctx[4]}"

  if [[ "$BACKGROUND" == "1" ]]; then
    mkdir -p "$eval_root"
    local log_path="${eval_root}/nohup_eval_api_$(timestamp).log"
    echo "Running in background:"
    echo "  $(join_cmd "${RUN_CMD[@]}")"
    nohup "${RUN_CMD[@]}" > "$log_path" 2>&1 &
    echo "Started PID: $!"
    echo "Log file: $log_path"
  else
    echo "Running:"
    echo "  $(join_cmd "${RUN_CMD[@]}")"
    "${RUN_CMD[@]}"
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
    --base-url|--base_url)
      BASE_URL="${2:-}"; shift 2 ;;
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
    --json-mode)
      JSON_MODE="${2:-}"; shift 2 ;;
    --cot-eval-mode)
      COT_EVAL_MODE="${2:-}"; shift 2 ;;
    --pipeline-mode)
      PIPELINE_MODE="${2:-}"; shift 2 ;;
    --postprocess-profile|--postprocess_profile)
      POSTPROCESS_PROFILE="${2:-}"; shift 2 ;;
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

if [[ "$JSON_MODE" != "auto" && "$JSON_MODE" != "on" && "$JSON_MODE" != "off" ]]; then
  echo "ERROR: --json-mode must be auto, on, or off."
  exit 1
fi
if [[ "$COT_EVAL_MODE" != "self_consistency" && "$COT_EVAL_MODE" != "counterfactual" ]]; then
  echo "ERROR: --cot-eval-mode must be self_consistency or counterfactual."
  exit 1
fi
if [[ "$PIPELINE_MODE" != "e2e" && "$PIPELINE_MODE" != "cat_lite" && "$PIPELINE_MODE" != "record_corrector" && "$PIPELINE_MODE" != "record_corrector+cat_lite" ]]; then
  echo "ERROR: --pipeline-mode must be e2e, cat_lite, record_corrector, or record_corrector+cat_lite."
  exit 1
fi
if [[ "$POSTPROCESS_PROFILE" != "none" && "$POSTPROCESS_PROFILE" != "event_probe_v2" ]]; then
  echo "ERROR: --postprocess-profile must be none or event_probe_v2."
  exit 1
fi

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
