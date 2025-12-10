#!/bin/bash
#SBATCH --job-name=llada_eval_quick
#SBATCH --output=logs/llada_eval_%A_%a.out
#SBATCH --error=logs/llada_eval_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-8%1

# Environment setup
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=/home/qiheng/Projects/adaptive-dllm:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2

# Activate environment
source ~/miniconda3/bin/activate adaptive-dllm

cd /home/qiheng/Projects/adaptive-dllm/evaluation/llada

# Create logs directory
mkdir -p logs results

# Model configuration
MODEL_PATH="GSAI-ML/LLaDA-8B-Base"
MODEL_TYPES=("standard" "sparse" "adaptive")

# Task configuration - PPL tasks (likelihood-based)
# Format: task_name:num_fewshot:mc_num:cfg
PPL_TASKS=(
    "mmlu:5:1:0.0"           # MMLU with 5-shot, mc_num=1, cfg=0.0
    "arc_challenge:0:128:0.5" # ARC-C with 0-shot, mc_num=128, cfg=0.5
    "hellaswag:0:128:0.5"     # HellaSwag with 0-shot, mc_num=128, cfg=0.5
    "truthfulqa_mc2:0:128:2.0" # TruthfulQA with 0-shot, mc_num=128, cfg=2.0
    "winogrande:5:128:0.0"    # WinoGrande with 5-shot, mc_num=128, cfg=0.0
    "piqa:0:128:0.5"          # PIQA with 0-shot, mc_num=128, cfg=0.5
)

# Task configuration - Generation tasks
# Format: task_name:gen_length:steps:block_length
GEN_TASKS=(
    "gsm8k:1024:1024:1024"    # GSM8K: gen_length:steps:block_length
    "minerva_math:1024:1024:1024" # Math
    "bbh:1024:1024:1024"      # BBH
)

# Combine all tasks
ALL_TASKS=(
    "${PPL_TASKS[@]}"
    "${GEN_TASKS[@]}"
)

# Get task for this array job
TASK_CONFIG="${ALL_TASKS[$SLURM_ARRAY_TASK_ID]}"
IFS=':' read -r TASK PARAM1 PARAM2 PARAM3 <<< "$TASK_CONFIG"

echo "================================================"
echo "Job ID: $SLURM_JOB_ID, Array ID: $SLURM_ARRAY_TASK_ID"
echo "Task: $TASK"
echo "================================================"

# Function to calculate block_size based on block_length
get_block_size() {
    local block_length=$1
    if [ "$block_length" -eq 1024 ]; then
        echo 128
    elif [ "$block_length" -eq 512 ]; then
        echo 64
    elif [ "$block_length" -eq 128 ]; then
        echo 32
    else
        echo 128  # Default fallback
    fi
}

# Function to run evaluation for all three model types
run_all_models() {
    local task=$1
    local param1=$2
    local param2=$3
    local param3=$4
    local is_gen=$5
    
    for model_type in "${MODEL_TYPES[@]}"; do
        echo ""
        echo "Running $model_type model on $task..."
        
        OUTPUT_DIR="results/${model_type}/${task}"
        mkdir -p "$OUTPUT_DIR"
        
        if [ "$is_gen" = "true" ]; then
            # Generation task
            GEN_LENGTH=$param1
            STEPS=$param2
            BLOCK_LENGTH=$param3
            BLOCK_SIZE=$(get_block_size $BLOCK_LENGTH)
            
            echo "Generation params: gen_length=${GEN_LENGTH}, steps=${STEPS}, block_length=${BLOCK_LENGTH}, block_size=${BLOCK_SIZE}"
            
            python -m accelerate.commands.launch --num_processes=1 eval_llada.py \
                --model llada_eval \
                --model_args model_path="${MODEL_PATH}",model_type="${model_type}",gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},skip=0.2,select=0.5,block_size=${BLOCK_SIZE},base_sparsity=0.5 \
                --tasks "${task}" \
                --limit 10 \
                --output_path "${OUTPUT_DIR}/results.json" \
                --log_samples \
                2>&1 | tee "${OUTPUT_DIR}/eval.log"
        else
            # PPL task (likelihood)
            NUM_FEWSHOT=$param1
            MC_NUM=$param2
            CFG=$param3
            BLOCK_SIZE=128  # Default block_size for PPL tasks
            
            # Determine batch size based on mc_num (must be divisible)
            if [ "$MC_NUM" -eq 1 ]; then
                BATCH_SIZE=1
            else
                BATCH_SIZE=8
            fi
            
            echo "PPL params: num_fewshot=${NUM_FEWSHOT}, mc_num=${MC_NUM}, cfg=${CFG}, block_size=${BLOCK_SIZE}, batch_size=${BATCH_SIZE}"
            
            python -m accelerate.commands.launch --num_processes=1 eval_llada.py \
                --model llada_eval \
                --model_args model_path="${MODEL_PATH}",model_type="${model_type}",mc_num=${MC_NUM},cfg=${CFG},is_check_greedy=False,skip=0.2,select=0.5,block_size=${BLOCK_SIZE},base_sparsity=0.5 \
                --tasks "${task}" \
                --num_fewshot ${NUM_FEWSHOT} \
                --limit 10 \
                --batch_size ${BATCH_SIZE} \
                --output_path "${OUTPUT_DIR}/results.json" \
                --log_samples \
                2>&1 | tee "${OUTPUT_DIR}/eval.log"
        fi
        
        echo "Completed $model_type model on $task"
    done
}

# Determine if it's a generation task
IS_GEN="false"
if [[ "$TASK" == "gsm8k" ]] || [[ "$TASK" == "minerva_math" ]] || [[ "$TASK" == "bbh" ]]; then
    IS_GEN="true"
fi

# Run evaluation
run_all_models "$TASK" "$PARAM1" "$PARAM2" "$PARAM3" "$IS_GEN"

echo ""
echo "================================================"
echo "All evaluations completed for $TASK"
echo "================================================"

