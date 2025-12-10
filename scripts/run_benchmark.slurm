#!/bin/bash
#SBATCH --job-name=benchmark_attn
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00

# Benchmark different attention methods

echo "=========================================="
echo "Attention Methods Benchmark"
echo "=========================================="
echo "Starting job on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Time: $(date)"
echo ""

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
NUM_SAMPLES=10
BATCH_SIZE=4
SEQ_LEN=128
STEPS=128
BLOCK_LENGTH=32
BLOCK_SIZE=128

echo "=========================================="
echo "Configuration:"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Samples: $NUM_SAMPLES"
echo "Batch size: $BATCH_SIZE"
echo "Sequence length: $SEQ_LEN"
echo "Steps: $STEPS"
echo ""

# Run benchmark comparing all three methods
echo "=========================================="
echo "Running Benchmark (All Methods)"
echo "=========================================="

python tests/test_benchmark_attention.py \
    --model_path $MODEL_PATH \
    --num_samples $NUM_SAMPLES \
    --batch_size $BATCH_SIZE \
    --seq_len $SEQ_LEN \
    --steps $STEPS \
    --block_length $BLOCK_LENGTH \
    --block_size $BLOCK_SIZE \
    --test_all \
    --sparse_keep_ratio 0.3 \
    --adaptive_avg_keep 0.3 \
    --output tests/benchmark_results_all.json \
    --seed 42

echo ""
echo "=========================================="
echo "Benchmark completed!"
echo "=========================================="
echo "Results saved to tests/benchmark_results_all.json"
echo ""
echo "Finished at $(date)"

