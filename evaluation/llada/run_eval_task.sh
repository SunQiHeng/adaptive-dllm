#!/bin/bash
# Quick test script for GSM8K and HumanEval
# Tests 3 model types on 2 tasks with reduced parameters
# Usage: bash run_eval_quick_test.sh

# Environment setup
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=/home/qiheng/Projects/adaptive-dllm:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# Activate environment
source ~/miniconda3/bin/activate adaptive-dllm

cd /home/qiheng/Projects/adaptive-dllm/evaluation/llada

# Create logs directory
mkdir -p logs results

echo "========================================================"
echo "Quick Test Configuration"
echo "========================================================"
echo "Tasks: GSM8K, HumanEval"
echo "Model Types: standard, sparse, adaptive"
echo "Generation Length: 256 tokens"
echo "Block Size: 32"
echo "Test Samples: 50 per dataset"
echo "========================================================"
echo ""

# Model configuration
MODEL_PATH="GSAI-ML/LLaDA-8B-Base"
MODEL_TYPES=("adaptive")

# Generation parameters
GEN_LENGTH=256
STEPS=256
BLOCK_LENGTH=32
BLOCK_SIZE=32
LIMIT=50

# Tasks to run
TASKS=("humaneval")

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
    
    echo "Params: gen_length=${GEN_LENGTH}, steps=${STEPS}, block_length=${BLOCK_LENGTH}, block_size=${BLOCK_SIZE}, limit=${LIMIT}"
    
    # Set importance source for adaptive mode
    if [ "$model_type" = "adaptive" ]; then
        IMPORTANCE_ARG=",importance_source=precomputed"
    else
        IMPORTANCE_ARG=""
    fi
    
    # Build the command based on task
    if [ "$task" = "humaneval" ]; then
        # HumanEval requires --confirm_run_unsafe_code
        python -m accelerate.commands.launch --num_processes=1 eval_llada.py \
            --model llada_eval \
            --model_args model_path="${MODEL_PATH}",model_type="${model_type}",gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},skip=0.2,select=0.3,block_size=${BLOCK_SIZE},base_sparsity=0.5${IMPORTANCE_ARG} \
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
            --model_args model_path="${MODEL_PATH}",model_type="${model_type}",gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},skip=0.2,select=0.3,block_size=${BLOCK_SIZE},base_sparsity=0.5${IMPORTANCE_ARG} \
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
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ${model_type} - ${task} - ${ELAPSED}s (${ELAPSED_MIN}m ${ELAPSED_SEC}s)" >> "results/timing_log.txt"
    
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

