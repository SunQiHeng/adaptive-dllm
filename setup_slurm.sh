#!/bin/bash

# Setup script for Adaptive Sparse Attention Slurm jobs
# This script prepares the environment for running slurm jobs

echo "=========================================="
echo "Adaptive Sparse Attention Setup"
echo "=========================================="
echo ""

# Create logs directory if it doesn't exist
if [ ! -d "logs" ]; then
    echo "Creating logs directory..."
    mkdir -p logs
    echo "✓ logs/ directory created"
else
    echo "✓ logs/ directory already exists"
fi

# Make slurm scripts executable
echo ""
echo "Making slurm scripts executable..."
chmod +x run_test_adaptive_sparse.slurm
chmod +x run_adaptive_simple.slurm
chmod +x run_adaptive_sparsed_generation.slurm
echo "✓ Slurm scripts are now executable"

# Check conda environment
echo ""
echo "Checking conda environment..."
if conda env list | grep -q "qwen3"; then
    echo "✓ Conda environment 'qwen3' exists"
else
    echo "⚠ Warning: Conda environment 'qwen3' not found"
    echo "  Please create it or modify the slurm scripts to use your environment"
fi

# Check Python and PyTorch
echo ""
echo "Checking Python packages..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwen3 2>/dev/null

if command -v python &> /dev/null; then
    echo "✓ Python: $(python --version 2>&1)"
    
    if python -c "import torch" 2>/dev/null; then
        echo "✓ PyTorch: $(python -c 'import torch; print(torch.__version__)')"
        
        if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
            echo "✓ CUDA available: Yes"
        else
            echo "⚠ CUDA available: No (required for generation tasks)"
        fi
    else
        echo "⚠ PyTorch not found"
    fi
    
    if python -c "import transformers" 2>/dev/null; then
        echo "✓ Transformers: $(python -c 'import transformers; print(transformers.__version__)')"
    else
        echo "⚠ Transformers not found"
    fi
else
    echo "⚠ Python not found"
fi

# Show available slurm scripts
echo ""
echo "=========================================="
echo "Available Slurm Scripts:"
echo "=========================================="
echo ""
echo "1. run_test_adaptive_sparse.slurm"
echo "   → Test adaptive sparse attention functionality (CPU only)"
echo "   → Usage: sbatch run_test_adaptive_sparse.slurm"
echo ""
echo "2. run_adaptive_simple.slurm"
echo "   → Quick test with default adaptive config (requires GPU)"
echo "   → Usage: sbatch run_adaptive_simple.slurm"
echo ""
echo "3. run_adaptive_sparsed_generation.slurm"
echo "   → Full test with 3 different configurations (requires GPU)"
echo "   → Usage: sbatch run_adaptive_sparsed_generation.slurm"
echo ""

# Show quick start commands
echo "=========================================="
echo "Quick Start Commands:"
echo "=========================================="
echo ""
echo "# Test functionality (no GPU needed):"
echo "sbatch run_test_adaptive_sparse.slurm"
echo ""
echo "# Quick generation test:"
echo "sbatch run_adaptive_simple.slurm"
echo ""
echo "# Check job status:"
echo "squeue -u \$USER"
echo ""
echo "# View output:"
echo "tail -f logs/adaptive_*.out"
echo ""

echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "For more information, see SLURM_USAGE.md"

