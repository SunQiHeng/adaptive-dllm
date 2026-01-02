#!/bin/bash
# Quick test script for GSM8K and HumanEval
# Tests 3 model types on 2 tasks with reduced parameters
# Usage: bash run_eval_quick_test.sh

# Environment setup
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=/home/qiheng/Projects/adaptive-dllm:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=5

# Activate environment
source ~/miniconda3/bin/activate adaptive-dllm

cd /home/qiheng/Projects/adaptive-dllm/evaluation/llada

# Model configuration
MODEL_PATH=${MODEL_PATH:-"/data/qh_models/LLaDA-1.5"}
MODEL_NAME=${MODEL_NAME:-"llada-1_5"}
MODEL_TYPES=("adaptive" "sparse" "standard")

# Output root (all results go here)
RESULTS_ROOT="/home/qiheng/Projects/adaptive-dllm/evaluation/llada/${MODEL_NAME}_results"
mkdir -p "$RESULTS_ROOT"

# Model-specific head-importance (used for loss_gateIG variants)
#
# You can extend this mapping if you add more models.
# You may also override it directly via env: MODEL_IMPORTANCE_ROOT=/path/to/config_dir
MODEL_IMPORTANCE_ROOT=${MODEL_IMPORTANCE_ROOT:-""}
case "$MODEL_NAME" in
    llada-8b-instruct|LLaDA-8B-Instruct)
        MODEL_IMPORTANCE_ROOT="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada-8b-instruct_loss_gateIG"
        ;;
    llada-base|llada_base|llada|LLaDA-Base|LLaDA-base)
        MODEL_IMPORTANCE_ROOT="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base_loss_gateIG"
        ;;
    llada1.5|llada-1.5|llada-1_5|llada_1_5|LLaDA-1.5|LLaDA-1_5)
        MODEL_IMPORTANCE_ROOT="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada-1_5_loss_gateIG"
        ;;
    *)
        MODEL_IMPORTANCE_ROOT=""
        ;;
esac

# Which precomputed head-importance to use for adaptive mode:
# - margin:       /home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base_margin/head_importance.pt
# - target_logit: /home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base_target_logit/head_importance.pt
#
# Shuffle ablations (distribution preserved, head indices permuted within each layer):
# - margin_shuf
# - target_logit_shuf
#
IMPORTANCE_TAG=${IMPORTANCE_TAG:-"loss_gateIG_neg"}  # margin | target_logit | all_ones | margin_shuf | target_logit_shuf | loss_gateIG | loss_gateIG_neg
SHUFFLE_SEED=${SHUFFLE_SEED:-1234}
if [ "$IMPORTANCE_TAG" = "margin" ]; then
    PRECOMPUTED_IMPORTANCE_PATH="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base_margin/head_importance.pt"
elif [ "$IMPORTANCE_TAG" = "target_logit" ]; then
    PRECOMPUTED_IMPORTANCE_PATH="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base_target_logit/head_importance.pt"
elif [ "$IMPORTANCE_TAG" = "all_ones" ]; then
    PRECOMPUTED_IMPORTANCE_PATH="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base_all_ones/head_importance.pt"
elif [ "$IMPORTANCE_TAG" = "margin_shuf" ]; then
    SRC_IMPORTANCE_PATH="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base_margin/head_importance.pt"
    SHUF_DIR="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base_margin_shuf_seed${SHUFFLE_SEED}"
    PRECOMPUTED_IMPORTANCE_PATH="${SHUF_DIR}/head_importance.pt"
elif [ "$IMPORTANCE_TAG" = "target_logit_shuf" ]; then
    SRC_IMPORTANCE_PATH="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base_target_logit/head_importance.pt"
    SHUF_DIR="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base_target_logit_shuf_seed${SHUFFLE_SEED}"
    PRECOMPUTED_IMPORTANCE_PATH="${SHUF_DIR}/head_importance.pt"
elif [ "$IMPORTANCE_TAG" = "loss_gateIG" ]; then
    if [ -z "$MODEL_IMPORTANCE_ROOT" ]; then
        echo "ERROR: Unknown MODEL_NAME='${MODEL_NAME}' for IMPORTANCE_TAG=loss_gateIG."
        echo "Please set MODEL_NAME to one of: llada-8b-instruct, llada-1_5 (or extend the mapping in this script)."
        exit 2
    fi
    PRECOMPUTED_IMPORTANCE_PATH="${MODEL_IMPORTANCE_ROOT}/head_importance.pt"
elif [ "$IMPORTANCE_TAG" = "loss_gateIG_neg" ]; then
    if [ -z "$MODEL_IMPORTANCE_ROOT" ]; then
        echo "ERROR: Unknown MODEL_NAME='${MODEL_NAME}' for IMPORTANCE_TAG=loss_gateIG_neg."
        echo "Please set MODEL_NAME to one of: llada-8b-instruct, llada-1_5 (or extend the mapping in this script)."
        exit 2
    fi
    SRC_IMPORTANCE_PATH="${MODEL_IMPORTANCE_ROOT}/head_importance.pt"
    NEG_DIR="${MODEL_IMPORTANCE_ROOT}_neg"
    PRECOMPUTED_IMPORTANCE_PATH="${NEG_DIR}/head_importance.pt"
else
    echo "ERROR: Unknown IMPORTANCE_TAG='$IMPORTANCE_TAG'. Use 'margin', 'target_logit', 'all_ones', 'margin_shuf', 'target_logit_shuf', 'loss_gateIG', or 'loss_gateIG_neg'."
    exit 2
fi

# Auto-generate shuffled importance if needed
if [ "$IMPORTANCE_TAG" = "margin_shuf" ] || [ "$IMPORTANCE_TAG" = "target_logit_shuf" ]; then
    if [ ! -f "$PRECOMPUTED_IMPORTANCE_PATH" ]; then
        echo "🔀 Generating shuffled importance (seed=${SHUFFLE_SEED})..."
        python /home/qiheng/Projects/adaptive-dllm/evaluation/llada/generate_shuffled_importance.py \
            --in_pt "$SRC_IMPORTANCE_PATH" \
            --out_dir "$SHUF_DIR" \
            --seed "$SHUFFLE_SEED"
        if [ ! -f "$PRECOMPUTED_IMPORTANCE_PATH" ]; then
            echo "ERROR: Failed to generate shuffled importance at: $PRECOMPUTED_IMPORTANCE_PATH"
            exit 3
        fi
    else
        echo "🔀 Using existing shuffled importance: $PRECOMPUTED_IMPORTANCE_PATH"
    fi
fi

# Auto-generate negated importance if needed
if [ "$IMPORTANCE_TAG" = "loss_gateIG_neg" ]; then
    if [ ! -f "$PRECOMPUTED_IMPORTANCE_PATH" ]; then
        echo "➖ Generating negated importance..."
        python /home/qiheng/Projects/adaptive-dllm/evaluation/llada/generate_negated_importance.py \
            --in_pt "$SRC_IMPORTANCE_PATH" \
            --out_dir "$NEG_DIR"
        if [ ! -f "$PRECOMPUTED_IMPORTANCE_PATH" ]; then
            echo "ERROR: Failed to generate negated importance at: $PRECOMPUTED_IMPORTANCE_PATH"
            exit 3
        fi
    else
        echo "➖ Using existing negated importance: $PRECOMPUTED_IMPORTANCE_PATH"
    fi
fi

echo "========================================================"
echo "Quick Test Configuration"
echo "========================================================"
echo "Tasks: GSM8K, HumanEval"
echo "Model Types: standard, sparse, adaptive"
echo "Generation Length: 256 tokens"
echo "Block Size: 32"
echo "Test Samples: 50 per dataset"
echo "Importance tag: ${IMPORTANCE_TAG}"
echo "Importance file: ${PRECOMPUTED_IMPORTANCE_PATH}"
echo "========================================================"
echo ""

# Generation parameters
GEN_LENGTH=256
STEPS=256
BLOCK_LENGTH=32
BLOCK_SIZE=32
LIMIT=100

# Tasks to run
TASKS=("gsm8k" "humaneval")

# Function to run evaluation for one model type on one task
run_single_eval() {
    local task=$1
    local model_type=$2
    
    echo ""
    echo "========================================"
    echo "Running: ${model_type} on ${task}"
    echo "========================================"
    
    OUTPUT_DIR="${RESULTS_ROOT}/${model_type}/${task}_${IMPORTANCE_TAG}"
    mkdir -p "$OUTPUT_DIR"
    
    # Record start time
    START_TIME=$(date +%s)
    
    echo "Params: gen_length=${GEN_LENGTH}, steps=${STEPS}, block_length=${BLOCK_LENGTH}, block_size=${BLOCK_SIZE}, limit=${LIMIT}"
    
    # Set importance source for adaptive mode
    if [ "$model_type" = "adaptive" ]; then
        IMPORTANCE_ARG=",importance_source=precomputed,precomputed_importance_path=${PRECOMPUTED_IMPORTANCE_PATH}"
    else
        IMPORTANCE_ARG=""
    fi
    
    # Build the command based on task
    if [ "$task" = "humaneval" ]; then
        # HumanEval requires --confirm_run_unsafe_code
        python -m accelerate.commands.launch --num_processes=1 eval_llada.py \
            --model llada_eval \
            --model_args model_path="${MODEL_PATH}",model_type="${model_type}",gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},skip=0.2,select=0.3,block_size=${BLOCK_SIZE}${IMPORTANCE_ARG} \
            --tasks "${task}" \
            --limit ${LIMIT} \
            --output_path "${OUTPUT_DIR}/results.json" \
            --log_samples \
            --confirm_run_unsafe_code \
            2>&1 | tee "${OUTPUT_DIR}/eval.log"
    else
        # GSM8K and other generation tasks
        python -m accelerate.commands.launch --num_processes=1 eval_llada.py \
            --model llada_eval \
            --model_args model_path="${MODEL_PATH}",model_type="${model_type}",gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},skip=0.2,select=0.3,block_size=${BLOCK_SIZE}${IMPORTANCE_ARG} \
            --tasks "${task}" \
            --limit ${LIMIT} \
            --output_path "${OUTPUT_DIR}/results.json" \
            --log_samples \
            2>&1 | tee "${OUTPUT_DIR}/eval.log"
    fi
    
    # Calculate running time
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))
    ELAPSED_SEC=$((ELAPSED % 60))
    
    # Record time to file
    echo "${ELAPSED}" > "${OUTPUT_DIR}/runtime.txt"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ${model_type} - ${task} - ${ELAPSED}s (${ELAPSED_MIN}m ${ELAPSED_SEC}s)" >> "${RESULTS_ROOT}/timing_log.txt"
    
    echo "✅ Completed ${model_type} on ${task}"
    echo "⏱️  Running time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s (${ELAPSED}s total)"
    echo ""
}

# Main execution
echo "🚀 Starting quick test evaluation..."
echo "Started at: $(date)"
echo ""

# Total tasks counter
TOTAL_TASKS=$((${#TASKS[@]} * ${#MODEL_TYPES[@]}))
CURRENT_TASK=0

# Run all combinations
for task in "${TASKS[@]}"; do
    echo ""
    echo "================================================"
    echo "📊 Task: ${task^^}"
    echo "================================================"
    
    for model_type in "${MODEL_TYPES[@]}"; do
        CURRENT_TASK=$((CURRENT_TASK + 1))
        echo ""
        echo "Progress: [${CURRENT_TASK}/${TOTAL_TASKS}]"
        
        run_single_eval "$task" "$model_type"
    done
done

echo ""
echo "================================================"
echo "✨ All evaluations completed!"
echo "Finished at: $(date)"
echo "================================================"
echo ""
echo "📁 Results saved in: ${RESULTS_ROOT}/"
echo "📊 Timing log: ${RESULTS_ROOT}/timing_log.txt"
echo ""

# Generate a summary
echo "📈 Summary:"
echo ""
for task in "${TASKS[@]}"; do
    echo "Task: ${task}"
    for model_type in "${MODEL_TYPES[@]}"; do
        RESULT_FILE="${RESULTS_ROOT}/${model_type}/${task}_${IMPORTANCE_TAG}/results.json"
        if [ -f "$RESULT_FILE" ]; then
            echo "  ✅ ${model_type}: ${RESULTS_ROOT}/${model_type}/${task}_${IMPORTANCE_TAG}/"
        else
            echo "  ❌ ${model_type}: FAILED"
        fi
    done
    echo ""
done

