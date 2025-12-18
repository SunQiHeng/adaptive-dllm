#!/bin/bash
# Quick test script for Dream with standard, sparse, and adaptive modes
# Tests 3 model types on 2 tasks with reduced parameters
# Usage: bash run_eval_quick_test.sh

# Environment setup
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=/home/qiheng/Projects/adaptive-dllm:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1

# Activate environment
source ~/miniconda3/bin/activate adaptive-dllm

cd /home/qiheng/Projects/adaptive-dllm/evaluation/dream

# Create logs directory
mkdir -p logs results

echo "========================================================"
echo "Dream Quick Test Configuration"
echo "========================================================"
echo "Tasks: GSM8K, HumanEval"
echo "Model Types: standard, sparse, adaptive"
echo "Max New Tokens: 256"
echo "Block Size: 32"
echo "Test Samples: 50 per dataset"
echo "========================================================"
echo ""

# Model configuration (matching attribution script)
MODEL_PATH="/data/qh_models/Dream-v0-Instruct-7B"
MODEL_TYPES=("adaptive" "sparse" "standard"  )

# Generation parameters (matching attribution script)
MAX_NEW_TOKENS=256
STEPS=32
TEMPERATURE=0.8
TOP_P=0.9
ALG="entropy"
ALG_TEMP=1.5
BLOCK_SIZE=32
LIMIT=50

# Sparse parameters
SKIP=0.2
SELECT=0.3

# Adaptive parameters
ADAPTIVE_STRATEGY="uniform"
BASE_SPARSITY=0.5
MIN_SPARSITY=0.1
MAX_SPARSITY=0.9

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
    
    OUTPUT_DIR="results/${model_type}/${task}"
    mkdir -p "$OUTPUT_DIR"
    
    # Record start time
    START_TIME=$(date +%s)
    
    echo "Params: max_new_tokens=${MAX_NEW_TOKENS}, steps=${STEPS}, block_size=${BLOCK_SIZE}, limit=${LIMIT}"
    
    # Build the command based on task
    if [ "$task" = "humaneval" ]; then
        # HumanEval requires --confirm_run_unsafe_code
        python -m accelerate.commands.launch --num_processes=1 eval_dream.py \
            --model dream_eval \
            --model_args model_path="${MODEL_PATH}",model_type="${model_type}",max_new_tokens=${MAX_NEW_TOKENS},steps=${STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},alg="${ALG}",alg_temp=${ALG_TEMP},skip=${SKIP},select=${SELECT},block_size=${BLOCK_SIZE},adaptive_strategy="${ADAPTIVE_STRATEGY}",base_sparsity=${BASE_SPARSITY},min_sparsity=${MIN_SPARSITY},max_sparsity=${MAX_SPARSITY} \
            --tasks "${task}" \
            --num_fewshot 0 \
            --limit ${LIMIT} \
            --output_path "${OUTPUT_DIR}/results.json" \
            --log_samples \
            --apply_chat_template \
            --confirm_run_unsafe_code \
            2>&1 | tee "${OUTPUT_DIR}/eval.log"
    else
        # GSM8K and other generation tasks
        python -m accelerate.commands.launch --num_processes=1 eval_dream.py \
            --model dream_eval \
            --model_args model_path="${MODEL_PATH}",model_type="${model_type}",max_new_tokens=${MAX_NEW_TOKENS},steps=${STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},alg="${ALG}",alg_temp=${ALG_TEMP},skip=${SKIP},select=${SELECT},block_size=${BLOCK_SIZE},adaptive_strategy="${ADAPTIVE_STRATEGY}",base_sparsity=${BASE_SPARSITY},min_sparsity=${MIN_SPARSITY},max_sparsity=${MAX_SPARSITY} \
            --tasks "${task}" \
            --num_fewshot 0 \
            --limit ${LIMIT} \
            --output_path "${OUTPUT_DIR}/results.json" \
            --log_samples \
            --apply_chat_template \
            2>&1 | tee "${OUTPUT_DIR}/eval.log"
    fi
    
    # Calculate running time
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))
    ELAPSED_SEC=$((ELAPSED % 60))
    
    # Record time to file
    echo "${ELAPSED}" > "${OUTPUT_DIR}/runtime.txt"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ${model_type} - ${task} - ${ELAPSED}s (${ELAPSED_MIN}m ${ELAPSED_SEC}s)" >> "results/timing_log.txt"
    
    echo "✅ Completed ${model_type} on ${task}"
    echo "⏱️  Running time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s (${ELAPSED}s total)"
    echo ""
}

# Main execution
echo "🚀 Starting Dream quick test evaluation..."
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
echo "📁 Results saved in: results/"
echo "📊 Timing log: results/timing_log.txt"
echo ""

# Generate a summary
echo "📈 Summary:"
echo ""
for task in "${TASKS[@]}"; do
    echo "Task: ${task}"
    for model_type in "${MODEL_TYPES[@]}"; do
        RESULT_FILE="results/${model_type}/${task}/results.json"
        if [ -f "$RESULT_FILE" ]; then
            echo "  ✅ ${model_type}: results/${model_type}/${task}/"
        else
            echo "  ❌ ${model_type}: FAILED"
        fi
    done
    echo ""
done

