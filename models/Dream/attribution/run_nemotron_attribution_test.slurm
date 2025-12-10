#!/bin/bash
#SBATCH --job-name=dream_attr_test
#SBATCH --output=logs/attribution_test_%j.out
#SBATCH --error=logs/attribution_test_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --partition=gpu

# Create logs directory
mkdir -p logs

# Configuration - TEST VERSION (small samples)
MODEL_PATH="/data/qh_models/Dream-v0-Instruct-7B"
SAMPLES_PER_CATEGORY=1  # Only 1 sample for quick test
MAX_NEW_TOKENS=64        # Shorter generation
STEPS=16                 # Fewer diffusion steps
STEP_INTERVAL=8          # Larger interval
N_ATTRIBUTION_STEPS=5    # Fewer attribution steps
BASE_OUTPUT_DIR="/home/qiheng/Projects/adaptive-dllm/models/Dream/attribution/attribution_results/test_run_$(date +%Y%m%d_%H%M%S)"

# Single seed for test
SEED=42

echo "=========================================="
echo "Dream Head Attribution - TEST RUN"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Model: $MODEL_PATH"
echo "Samples per category: $SAMPLES_PER_CATEGORY (TEST)"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Diffusion steps: $STEPS"
echo "Step interval: $STEP_INTERVAL"
echo "Attribution steps: $N_ATTRIBUTION_STEPS"
echo "Output dir: $BASE_OUTPUT_DIR"
echo "Seed: $SEED"
echo "=========================================="

# Navigate to project root
cd /home/qiheng/Projects/adaptive-dllm

echo ""
echo "Starting test run..."
echo "Start time: $(date)"
echo ""

python models/Dream/attribution/compute_nemotron_attribution_stepwise_dream.py \
    --model_path "$MODEL_PATH" \
    --samples_per_category $SAMPLES_PER_CATEGORY \
    --max_new_tokens $MAX_NEW_TOKENS \
    --steps $STEPS \
    --step_interval $STEP_INTERVAL \
    --n_attribution_steps $N_ATTRIBUTION_STEPS \
    --output_dir "$BASE_OUTPUT_DIR" \
    --seed $SEED \
    --alg_temp 0.0 \
    --temperature 0.8 \
    --top_p 0.9 \
    --alg entropy \
    --device cuda

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "TEST RUN COMPLETED SUCCESSFULLY!"
    echo "Results saved to: $BASE_OUTPUT_DIR"
    echo ""
    echo "Check the results:"
    echo "  ls -lh $BASE_OUTPUT_DIR"
    echo ""
    echo "If test looks good, run full version with:"
    echo "  sbatch models/Dream/attribution/run_nemotron_attribution.slurm"
else
    echo "TEST RUN FAILED with exit code $EXIT_CODE"
    echo "Check logs for errors"
fi
echo "=========================================="
echo "End time: $(date)"

