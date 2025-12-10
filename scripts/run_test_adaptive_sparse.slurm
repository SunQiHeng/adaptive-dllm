#!/bin/bash
#SBATCH --job-name=test_adaptive
#SBATCH --output=logs/test_adaptive_%j.out
#SBATCH --error=logs/test_adaptive_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00

# Print job info
echo "=========================================="
echo "Adaptive Sparse Attention Test Suite"
echo "=========================================="
echo "Starting job on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo ""

# Change to the project directory
cd /home/qiheng/Projects/adaptive-dllm

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3

# Print Python and PyTorch versions
echo "Python version:"
python --version
echo ""
echo "PyTorch version:"
python -c "import torch; print(torch.__version__)"
echo ""

# Run the test script
echo "=========================================="
echo "Running Adaptive Sparse Attention Tests"
echo "=========================================="
python tests/test_adaptive_sparse.py

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ All tests passed successfully!"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ Some tests failed!"
    echo "=========================================="
fi

echo ""
echo "Job finished at $(date)"

