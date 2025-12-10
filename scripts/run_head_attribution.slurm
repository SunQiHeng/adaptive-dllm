#!/bin/bash
#SBATCH --job-name=head_attr
#SBATCH --output=logs/head_attr_%j.out
#SBATCH --error=logs/head_attr_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3

# 切换到工作目录
cd /home/qiheng/Projects/adaptive-dllm

# 运行 head attribution 测试
echo "Starting head attribution test..."
python examples/compute_head_importance_v2.py

echo "Test completed!"

