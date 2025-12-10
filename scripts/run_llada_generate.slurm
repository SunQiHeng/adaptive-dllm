#!/bin/bash
#SBATCH --job-name=llada_gen
#SBATCH --output=logs/llada_gen_%j.out
#SBATCH --error=logs/llada_gen_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --partition=gpu

# Print job info
echo "Starting job on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo ""

# Change to the project directory
cd /home/qiheng/Projects/adaptive-dllm

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3

# Run the script
python models/LLaDA/generation/generate.py

echo ""
echo "Job finished at $(date)"

