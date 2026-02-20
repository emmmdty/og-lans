#!/usr/bin/env bash
# ==============================================================================
# OG-LANS 快速调试脚本 (30分钟内完成全流程验证)
# ==============================================================================
#
# 用途: 快速验证代码是否能正常运行，不追求训练效果
# 预计耗时: 
#   - 模型加载: 2-3 分钟
#   - 数据预处理: 1-2 分钟  
#   - 训练 20 步: 10-15 分钟
#   - 评估 10 样本: 3-5 分钟
#   - 总计: ~20-30 分钟
#
# 使用方法:
#   chmod +x scripts/run_debug.sh
#   ./scripts/run_debug.sh
#   ./scripts/run_debug.sh --quick
#   ./scripts/run_debug.sh --skip-tests
#   ./scripts/run_debug.sh --skip-eval
#   ./scripts/run_debug.sh --allow-test-fail
#
# ==============================================================================

set -euo pipefail

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 默认参数
SKIP_TESTS=false
SKIP_TRAINING=false
SKIP_EVAL=false
MAX_STEPS=20
EVAL_SAMPLES=10
QUICK_MODE=false
ALLOW_TEST_FAIL=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-tests)
            SKIP_TESTS=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-eval)
            SKIP_EVAL=true
            shift
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --eval-samples)
            EVAL_SAMPLES="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            MAX_STEPS=5
            EVAL_SAMPLES=5
            shift
            ;;
        --allow-test-fail)
            ALLOW_TEST_FAIL=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-tests      跳过单元测试"
            echo "  --skip-training   跳过训练"
            echo "  --skip-eval       跳过评估"
            echo "  --quick           快速冒烟模式 (= --max-steps 5 --eval-samples 5)"
            echo "  --allow-test-fail 单测失败后继续执行（默认失败即退出）"
            echo "  --max-steps N     最大训练步数 (默认: 20)"
            echo "  --eval-samples N  评估样本数 (默认: 10)"
            echo "  -h, --help        显示帮助"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"
PROJECT_ROOT="$ROOT_DIR"

START_TIME=$(date +%s)

echo -e "${CYAN}===================================${NC}"
echo -e "${CYAN}OG-LANS${NC}"
echo -e "${CYAN}===================================${NC}"
echo "Started at: $(date)"
echo "Project root: $PROJECT_ROOT"
if [ "$QUICK_MODE" = true ]; then
    echo "Mode: QUICK (max_steps=$MAX_STEPS, eval_samples=$EVAL_SAMPLES)"
fi
echo ""

# 设置环境变量
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export WANDB_MODE="offline"
export PYTHONUNBUFFERED="1"
resolve_python_bin() {
    if command -v python &> /dev/null; then
        echo "python"
        return
    fi
    if command -v python3 &> /dev/null; then
        echo "python3"
        return
    fi
    echo ""
}
PYTHON_BIN="$(resolve_python_bin)"

# 检查 Python 环境
echo -e "${YELLOW}Checking Python environment...${NC}"
if [[ -n "$PYTHON_BIN" ]]; then
    PYTHON_VERSION=$("$PYTHON_BIN" --version 2>&1)
    echo -e "  Python: ${GREEN}$PYTHON_VERSION${NC}"
else
    echo -e "${RED}ERROR: Neither python nor python3 found!${NC}"
    exit 1
fi

# 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo -e "  GPU: ${GREEN}$GPU_INFO${NC}"
else
    echo -e "  GPU: ${YELLOW}nvidia-smi not found, CPU mode${NC}"
fi

# Step 1: 单元测试
echo ""
if [ "$SKIP_TESTS" = false ]; then
    echo -e "${YELLOW}[Step 1/4] Running Unit Tests...${NC}"
    echo "-----------------------------------"
    
    if "$PYTHON_BIN" -m pytest tests/test_lans.py tests/test_scv.py -v --tb=short -q; then
        echo -e "${GREEN}SUCCESS: All tests passed${NC}"
    else
        if [ "$ALLOW_TEST_FAIL" = true ]; then
            echo -e "${YELLOW}WARNING: Some tests failed, but --allow-test-fail is set. Continuing...${NC}"
        else
            echo -e "${RED}ERROR: Unit tests failed. Stop here to avoid invalid debug conclusions.${NC}"
            echo "Hint: use --allow-test-fail if you intentionally want to continue."
            exit 2
        fi
    fi
else
    echo -e "[Step 1/4] Unit Tests ${YELLOW}SKIPPED${NC}"
fi

# Step 2: 训练
echo ""
if [ "$SKIP_TRAINING" = false ]; then
    echo -e "${YELLOW}[Step 2/4] Starting Debug Training...${NC}"
    echo "-----------------------------------"
    echo "  Config: configs/config_debug.yaml"
    echo "  Max samples: 50"
    echo "  Max steps: $MAX_STEPS"
    echo "  Heartbeat keywords: 初始负样本进度 / Epoch 1 负样本刷新进度 / SCV 心跳"
    echo ""
    
    TRAIN_START=$(date +%s)

    if "$PYTHON_BIN" main.py --config configs/config_debug.yaml --training.max_steps "$MAX_STEPS"; then
        TRAIN_END=$(date +%s)
        TRAIN_DURATION=$((TRAIN_END - TRAIN_START))
        echo -e "${GREEN}SUCCESS: Training completed in $((TRAIN_DURATION / 60)) min $((TRAIN_DURATION % 60)) sec${NC}"
    else
        echo -e "${RED}ERROR: Training failed!${NC}"
        echo "Check the logs above for details."
        exit 1
    fi
else
    echo -e "[Step 2/4] Training ${YELLOW}SKIPPED${NC}"
fi

# Step 3: 检查 Checkpoint
echo ""
echo -e "${YELLOW}[Step 3/4] Checking Checkpoint...${NC}"
echo "-----------------------------------"

CHECKPOINT_DIR="logs/debug/checkpoints"
LATEST_CKPT=""

if [ -d "$CHECKPOINT_DIR" ]; then
    # 首先检查根目录是否直接包含模型文件（save_strategy: no 的情况）
    if [ -f "$CHECKPOINT_DIR/adapter_config.json" ] || [ -f "$CHECKPOINT_DIR/adapter_model.safetensors" ]; then
        LATEST_CKPT="$CHECKPOINT_DIR"
        echo -e "${GREEN}SUCCESS: Found checkpoint in root directory${NC}"
        echo "  Path: $LATEST_CKPT"
        # 列出模型文件
        ls -la "$CHECKPOINT_DIR"/*.json "$CHECKPOINT_DIR"/*.safetensors 2>/dev/null | head -5
    else
        # 查找 checkpoint-* 子目录（save_strategy: steps 的情况）
        LATEST_CKPT=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | sort -V | tail -1)
        
        if [ -n "$LATEST_CKPT" ]; then
            CKPT_COUNT=$(find "$CHECKPOINT_DIR" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | wc -l)
            echo -e "${GREEN}SUCCESS: Found $CKPT_COUNT checkpoint(s)${NC}"
            echo "  Latest: $(basename "$LATEST_CKPT")"
        else
            echo -e "${YELLOW}WARNING: No checkpoint found in $CHECKPOINT_DIR${NC}"
            echo "  Directory contents:"
            ls -la "$CHECKPOINT_DIR" 2>/dev/null || echo "  (empty or not accessible)"
        fi
    fi
else
    echo -e "${YELLOW}WARNING: Checkpoint directory not found${NC}"
fi

# Step 4: 评估
echo ""
if [ "$SKIP_EVAL" = false ] && [ -n "$LATEST_CKPT" ]; then
    echo -e "${YELLOW}[Step 4/4] Running Quick Evaluation...${NC}"
    echo "-----------------------------------"
    echo "  Checkpoint: $LATEST_CKPT"
    echo "  Samples: $EVAL_SAMPLES"
    echo ""
    
    EVAL_START=$(date +%s)
    
    if "$PYTHON_BIN" evaluate.py --config configs/config_debug.yaml --checkpoint "$LATEST_CKPT" --num_samples "$EVAL_SAMPLES"; then
        EVAL_END=$(date +%s)
        EVAL_DURATION=$((EVAL_END - EVAL_START))
        echo -e "${GREEN}SUCCESS: Evaluation completed in $((EVAL_DURATION / 60)) min $((EVAL_DURATION % 60)) sec${NC}"
    else
        echo -e "${YELLOW}WARNING: Evaluation encountered errors${NC}"
    fi
else
    echo -e "[Step 4/4] Evaluation ${YELLOW}SKIPPED${NC}"
    if [ -z "$LATEST_CKPT" ]; then
        echo "  Reason: No checkpoint available"
    fi
fi

# 总结
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo ""
echo -e "${CYAN}===================================${NC}"
echo -e "${CYAN}Debug Flow Complete!${NC}"
echo -e "${CYAN}===================================${NC}"
echo ""
echo "Summary:"
echo "  Total time: $((TOTAL_DURATION / 60)) min $((TOTAL_DURATION % 60)) sec"
echo "  Unit tests: $([ "$SKIP_TESTS" = true ] && echo 'Skipped' || echo 'Executed')"
echo "  Training: $([ "$SKIP_TRAINING" = true ] && echo 'Skipped' || echo 'Completed')"
echo "  Checkpoint: $([ -n "$LATEST_CKPT" ] && basename "$LATEST_CKPT" || echo 'Not found')"
echo "  Evaluation: $([ "$SKIP_EVAL" = true ] || [ -z "$LATEST_CKPT" ] && echo 'Skipped' || echo 'Completed')"
echo ""
echo -e "${GREEN}If all steps passed without errors, your code is ready!${NC}"
echo ""
echo "Log locations:"
echo "  Checkpoints: logs/debug/checkpoints/"
echo "  TensorBoard: logs/debug/tensorboard/"
echo "  Filtered samples: logs/debug/samples/"
echo ""
echo "To visualize training, run:"
echo "  tensorboard --logdir logs/debug/tensorboard"
echo ""
echo "To clean up debug logs:"
echo "  rm -rf logs/debug/"
