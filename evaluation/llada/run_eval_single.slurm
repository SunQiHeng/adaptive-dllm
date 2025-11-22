#!/bin/bash
#SBATCH --job-name=llada_eval_single
#SBATCH --output=logs/llada_eval_single_%j.out
#SBATCH --error=logs/llada_eval_single_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# Usage: sbatch run_eval_single.slurm [model_type] [task] [extra_args]
# Example: sbatch run_eval_single.slurm adaptive gsm8k

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=/home/qiheng/Projects/adaptive-dllm:$PYTHONPATH

cd /home/qiheng/Projects/adaptive-dllm/evaluation/llada
mkdir -p logs results

MODEL_PATH="GSAI-ML/LLaDA-8B-Base"
MODEL_TYPE=${1:-"standard"}  # standard, sparse, or adaptive
TASK=${2:-"gsm8k"}

echo "================================================"
echo "LLaDA Evaluation - Single Task"
echo "Model Type: $MODEL_TYPE"
echo "Task: $TASK"
echo "================================================"

OUTPUT_DIR="results/${MODEL_TYPE}/${TASK}"
mkdir -p "$OUTPUT_DIR"

# Default parameters for common tasks
case $TASK in
    gsm8k|minerva_math|bbh)
        # Generation tasks
        accelerate launch --num_processes=1 eval_llada.py \
            --model llada_eval \
            --model_args model_path="${MODEL_PATH}",model_type="${MODEL_TYPE}",gen_length=1024,steps=1024,block_length=1024,skip=0.2,select=0.3,block_size=128,base_sparsity=0.5 \
            --tasks "${TASK}" \
            --output_path "${OUTPUT_DIR}/results.json" \
            --log_samples
        ;;
    mmlu)
        # PPL task - MMLU (special: mc_num=1)
        accelerate launch --num_processes=1 eval_llada.py \
            --model llada_eval \
            --model_args model_path="${MODEL_PATH}",model_type="${MODEL_TYPE}",mc_num=1,cfg=0.0,is_check_greedy=False \
            --tasks "${TASK}" \
            --num_fewshot 5 \
            --batch_size 1 \
            --output_path "${OUTPUT_DIR}/results.json" \
            --log_samples
        ;;
    *)
        # Other PPL tasks
        accelerate launch --num_processes=1 eval_llada.py \
            --model llada_eval \
            --model_args model_path="${MODEL_PATH}",model_type="${MODEL_TYPE}",mc_num=128,cfg=0.5,is_check_greedy=False \
            --tasks "${TASK}" \
            --num_fewshot 0 \
            --batch_size 8 \
            --output_path "${OUTPUT_DIR}/results.json" \
            --log_samples
        ;;
esac

echo ""
echo "================================================"
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR/results.json"
echo "================================================"

