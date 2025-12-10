#!/bin/bash
#SBATCH --job-name=adaptive_quick
#SBATCH --output=logs/adaptive_quick_%j.out
#SBATCH --error=logs/adaptive_quick_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# Quick test for Adaptive Sparse Attention

echo "作业开始: $(date)"
echo "节点: $SLURM_JOB_NODELIST"
echo "Job ID: $SLURM_JOB_ID"
echo ""

cd /home/qiheng/Projects/adaptive-dllm

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3

echo "=========================================="
echo "Quick Adaptive Sparse Attention Test"
echo "=========================================="

# Run with default adaptive configuration
python models/LLaDA/generation/adaptive_sparsed_generate.py \
    --model_path GSAI-ML/LLaDA-1.5 \
    --use_adaptive \
    --seq_len 128 \
    --steps 128 \
    --block_length 32 \
    --verbose

echo ""
echo "作业完成: $(date)"

