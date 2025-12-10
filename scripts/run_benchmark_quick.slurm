#!/bin/bash
#SBATCH --job-name=bench_quick
#SBATCH --output=logs/benchmark_quick_%j.out
#SBATCH --error=logs/benchmark_quick_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:45:00

# Quick benchmark (fewer samples for faster testing)

echo "=========================================="
echo "Quick Attention Methods Benchmark"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo ""

cd /home/qiheng/Projects/adaptive-dllm

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3

# Quick test with fewer samples
python tests/test_benchmark_attention.py \
    --model_path GSAI-ML/LLaDA-1.5 \
    --num_samples 6 \
    --batch_size 2 \
    --seq_len 128 \
    --steps 128 \
    --block_length 32 \
    --test_all \
    --sparse_keep_ratio 0.3 \
    --adaptive_avg_keep 0.3 \
    --output tests/benchmark_results_quick.json \
    --seed 42 \
    --print_outputs

echo ""
echo "Finished at $(date)"

