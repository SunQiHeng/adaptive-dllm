#!/bin/bash
#SBATCH --job-name=llada_eval
#SBATCH --output=logs/llada_eval_%A_%a.out
#SBATCH --error=logs/llada_eval_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-8

# Environment setup
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=/home/qiheng/Projects/adaptive-dllm:$PYTHONPATH

cd /home/qiheng/Projects/adaptive-dllm/evaluation/llada

# Create logs directory
mkdir -p logs results

# Model configuration
MODEL_PATH="GSAI-ML/LLaDA-8B-Base"
MODEL_TYPES=("standard" "sparse" "adaptive")

# Task configuration - PPL tasks (likelihood-based)
PPL_TASKS=(
    "mmlu:5:1:0.0"           # MMLU with 5-shot, mc_num=1, cfg=0
    "arc_challenge:0:128:0.5" # ARC-C with 0-shot, mc_num=128, cfg=0.5
    "hellaswag:0:128:0.5"     # HellaSwag
    "truthfulqa_mc2:0:128:2.0" # TruthfulQA
    "winogrande:5:128:0.0"    # WinoGrande
    "piqa:0:128:0.5"          # PIQA
)

# Task configuration - Generation tasks
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
            
            accelerate launch --num_processes=1 eval_llada.py \
                --model llada_eval \
                --model_args model_path="${MODEL_PATH}",model_type="${model_type}",gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},skip=0.2,select=0.3,block_size=128,base_sparsity=0.5 \
                --tasks "${task}" \
                --output_path "${OUTPUT_DIR}/results.json" \
                --log_samples \
                2>&1 | tee "${OUTPUT_DIR}/eval.log"
        else
            # PPL task (likelihood)
            NUM_FEWSHOT=$param1
            MC_NUM=$param2
            CFG=$param3
            
            accelerate launch --num_processes=1 eval_llada.py \
                --model llada_eval \
                --model_args model_path="${MODEL_PATH}",model_type="${model_type}",mc_num=${MC_NUM},cfg=${CFG},is_check_greedy=False,skip=0.2,select=0.3,block_size=128,base_sparsity=0.5 \
                --tasks "${task}" \
                --num_fewshot ${NUM_FEWSHOT} \
                --batch_size 8 \
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

