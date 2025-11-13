#!/bin/bash
#SBATCH --job-name=test_attr
#SBATCH --output=logs/test_attr_%j.out
#SBATCH --error=logs/test_attr_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# 激活 conda 环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3

# # 指定使用 GPU 2（或其他空闲的 GPU）
# export CUDA_VISIBLE_DEVICES=2

# 切换到工作目录
cd /home/qiheng/Projects/adaptive-dllm

# 运行测试
echo "Starting head attribution tests..."
python tests/test_head_attribution.py

echo "Tests completed!"

