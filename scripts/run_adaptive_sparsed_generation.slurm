#!/bin/bash
#SBATCH --job-name=adaptive_gen
#SBATCH --output=logs/adaptive_gen_%j.out
#SBATCH --error=logs/adaptive_gen_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# Print job info
echo "=========================================="
echo "Adaptive Sparse Attention Generation"
echo "=========================================="
echo "Starting job on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Time: $(date)"
echo ""

# Change to the project directory
cd /home/qiheng/Projects/adaptive-dllm

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3

# Print environment info
echo "Python version:"
python --version
echo ""
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")')"
echo ""

# Configuration
MODEL_PATH="GSAI-ML/LLaDA-1.5"
SEQ_LEN=128
STEPS=128
BLOCK_LENGTH=32
BLOCK_SIZE=128

# ========================================
# Test 1: Adaptive Sparse Attention (Conservative)
# ========================================
echo "=========================================="
echo "Test 1: Adaptive Sparse (Conservative)"
echo "=========================================="
echo "Configuration:"
echo "  - Base sparsity: 0.5"
echo "  - Sparsity range: [0.3, 0.7]"
echo "  - Strategy: uniform"
echo ""

python models/LLaDA/generation/adaptive_sparsed_generate.py \
    --model_path $MODEL_PATH \
    --use_adaptive \
    --seq_len $SEQ_LEN \
    --steps $STEPS \
    --block_length $BLOCK_LENGTH \
    --block_size $BLOCK_SIZE \
    --base_sparsity 0.5 \
    --min_sparsity 0.3 \
    --max_sparsity 0.7 \
    --importance_strategy uniform \
    --seed 42 \
    --verbose

echo ""
echo "Test 1 completed."
echo ""

# ========================================
# Test 2: Adaptive Sparse Attention (Aggressive)
# ========================================
echo "=========================================="
echo "Test 2: Adaptive Sparse (Aggressive)"
echo "=========================================="
echo "Configuration:"
echo "  - Base sparsity: 0.5"
echo "  - Sparsity range: [0.1, 0.95]"
echo "  - Strategy: exponential"
echo ""

python models/LLaDA/generation/adaptive_sparsed_generate.py \
    --model_path $MODEL_PATH \
    --use_adaptive \
    --seq_len $SEQ_LEN \
    --steps $STEPS \
    --block_length $BLOCK_LENGTH \
    --block_size $BLOCK_SIZE \
    --base_sparsity 0.5 \
    --min_sparsity 0.1 \
    --max_sparsity 0.95 \
    --importance_strategy exponential \
    --seed 42

echo ""
echo "Test 2 completed."
echo ""

# ========================================
# Test 3: Standard Sparse Attention (Comparison)
# ========================================
echo "=========================================="
echo "Test 3: Standard Sparse (Comparison)"
echo "=========================================="
echo "Configuration:"
echo "  - Skip: 0.2"
echo "  - Select: 0.3"
echo "  - Block size: 128"
echo ""

python models/LLaDA/generation/adaptive_sparsed_generate.py \
    --model_path $MODEL_PATH \
    --seq_len $SEQ_LEN \
    --steps $STEPS \
    --block_length $BLOCK_LENGTH \
    --skip 0.2 \
    --select 0.3 \
    --block_size $BLOCK_SIZE \
    --seed 42

echo ""
echo "Test 3 completed."
echo ""

# ========================================
# Summary
# ========================================
echo "=========================================="
echo "All tests completed!"
echo "=========================================="
echo "Finished at $(date)"
echo ""
echo "Results saved to:"
echo "  - Output: logs/adaptive_gen_$SLURM_JOB_ID.out"
echo "  - Errors: logs/adaptive_gen_$SLURM_JOB_ID.err"
echo "=========================================="

