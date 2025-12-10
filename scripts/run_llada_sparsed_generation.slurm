#!/bin/bash
#SBATCH --job-name=llada_gen
#SBATCH --output=logs/llada_gen_%j.out
#SBATCH --error=logs/llada_gen_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

echo "作业开始: $(date)"
echo "节点: $SLURM_JOB_NODELIST"

cd /home/qiheng/Projects/adaptive-dllm

# 运行 sparsed_generate.py (稀疏版本，默认参数)
python models/LLaDA/generation/sparsed_generate.py --skip 0.2 --select 0.3 --block_size 128 --prompt short_context

echo "作业完成: $(date)"

