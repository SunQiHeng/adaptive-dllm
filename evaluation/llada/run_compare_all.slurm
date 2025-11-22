#!/bin/bash
#SBATCH --job-name=llada_compare
#SBATCH --output=logs/llada_compare_%j.out
#SBATCH --error=logs/llada_compare_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00

# Compare all three attention types on a single task
# Usage: sbatch run_compare_all.slurm [task] [gen_length] [steps]
# Example: sbatch run_compare_all.slurm gsm8k 512 512

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=/home/qiheng/Projects/adaptive-dllm:$PYTHONPATH

cd /home/qiheng/Projects/adaptive-dllm/evaluation/llada
mkdir -p logs results

MODEL_PATH="GSAI-ML/LLaDA-8B-Base"
TASK=${1:-"gsm8k"}
GEN_LENGTH=${2:-512}
STEPS=${3:-512}
BLOCK_LENGTH=${4:-128}

echo "======================================================================="
echo "LLaDA Evaluation - Compare All Attention Types"
echo "======================================================================="
echo "Task: $TASK"
echo "Gen Length: $GEN_LENGTH, Steps: $STEPS, Block Length: $BLOCK_LENGTH"
echo "Model Path: $MODEL_PATH"
echo "======================================================================="
echo ""

# Define model types
MODEL_TYPES=("standard" "sparse" "adaptive")

# Determine if it's a generation task
case $TASK in
    gsm8k|minerva_math|bbh|humaneval|mbpp)
        IS_GEN=true
        ;;
    *)
        IS_GEN=false
        ;;
esac

# Run evaluation for each model type
for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
    echo ""
    echo "-----------------------------------------------------------------------"
    echo "Running: $MODEL_TYPE"
    echo "-----------------------------------------------------------------------"
    
    OUTPUT_DIR="results/compare/${TASK}/${MODEL_TYPE}"
    mkdir -p "$OUTPUT_DIR"
    
    START_TIME=$(date +%s)
    
    if [ "$IS_GEN" = true ]; then
        # Generation tasks
        echo "Running generation task..."
        accelerate launch --num_processes=1 eval_llada.py \
            --model llada_eval \
            --model_args model_path="${MODEL_PATH}",model_type="${MODEL_TYPE}",gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},skip=0.2,select=0.3,block_size=128,base_sparsity=0.5 \
            --tasks "${TASK}" \
            --output_path "${OUTPUT_DIR}/results.json" \
            --log_samples \
            2>&1 | tee "${OUTPUT_DIR}/eval.log"
    else
        # PPL tasks
        case $TASK in
            mmlu)
                echo "Running MMLU (PPL task)..."
                accelerate launch --num_processes=1 eval_llada.py \
                    --model llada_eval \
                    --model_args model_path="${MODEL_PATH}",model_type="${MODEL_TYPE}",mc_num=1,cfg=0.0,is_check_greedy=False,skip=0.2,select=0.3,block_size=128,base_sparsity=0.5 \
                    --tasks "${TASK}" \
                    --num_fewshot 5 \
                    --batch_size 1 \
                    --output_path "${OUTPUT_DIR}/results.json" \
                    --log_samples \
                    2>&1 | tee "${OUTPUT_DIR}/eval.log"
                ;;
            *)
                echo "Running PPL task..."
                accelerate launch --num_processes=1 eval_llada.py \
                    --model llada_eval \
                    --model_args model_path="${MODEL_PATH}",model_type="${MODEL_TYPE}",mc_num=128,cfg=0.5,is_check_greedy=False,skip=0.2,select=0.3,block_size=128,base_sparsity=0.5 \
                    --tasks "${TASK}" \
                    --num_fewshot 0 \
                    --batch_size 8 \
                    --output_path "${OUTPUT_DIR}/results.json" \
                    --log_samples \
                    2>&1 | tee "${OUTPUT_DIR}/eval.log"
                ;;
        esac
    fi
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo "✓ Completed $MODEL_TYPE in ${DURATION}s"
    echo "  Results: ${OUTPUT_DIR}/results.json"
    
    # Clear GPU cache
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
    sleep 5
done

echo ""
echo "======================================================================="
echo "All evaluations completed!"
echo "======================================================================="
echo ""
echo "View results:"
echo "  cd results/compare/${TASK}"
echo "  ls -lh */results.json"
echo ""
echo "Or run comparison script:"
echo "  python compare_results.py ${TASK}"
echo "======================================================================="

