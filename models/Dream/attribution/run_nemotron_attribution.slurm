#!/bin/bash
#SBATCH --job-name=dream_attribution_3runs
#SBATCH --output=logs/attribution_3runs_%j.out
#SBATCH --error=logs/attribution_3runs_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Create logs directory
mkdir -p logs

# Configuration
MODEL_PATH=${MODEL_PATH:-"/data/qh_models/Dream-v0-Instruct-7B"}
SAMPLES_PER_CATEGORY=${SAMPLES_PER_CATEGORY:-5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
STEPS=${STEPS:-32}
STEP_INTERVAL=${STEP_INTERVAL:-4}
N_ATTRIBUTION_STEPS=${N_ATTRIBUTION_STEPS:-10}
BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-"/home/qiheng/Projects/adaptive-dllm/models/Dream/attribution/attribution_results/run_$(date +%Y%m%d_%H%M%S)"}

# GPU will be assigned by SLURM, but we can set a preference
# export CUDA_VISIBLE_DEVICES will be set by SLURM automatically

# 定义三个不同的随机种子
SEEDS=(42 123 2024)

echo "=========================================="
echo "Dream Head Attribution - 3 Runs Comparison"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "Model: $MODEL_PATH"
echo "Samples per category: $SAMPLES_PER_CATEGORY"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Diffusion steps: $STEPS"
echo "Step interval: $STEP_INTERVAL"
echo "Attribution steps: $N_ATTRIBUTION_STEPS"
echo "Base output dir: $BASE_OUTPUT_DIR"
echo "Seeds: ${SEEDS[@]}"
echo "=========================================="

# Navigate to project root
cd /home/qiheng/Projects/adaptive-dllm

# Activate environment if needed
# source /path/to/venv/bin/activate

# Run attribution 3 times with different seeds
for i in {0..2}; do
    SEED=${SEEDS[$i]}
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/run_$((i+1))_seed_${SEED}"
    
    echo ""
    echo "=========================================="
    echo "Run $((i+1))/3 - Seed: $SEED"
    echo "=========================================="
    echo "Output: $OUTPUT_DIR"
    echo "Start time: $(date)"
    echo ""
    
    python models/Dream/attribution/compute_nemotron_attribution_stepwise_dream.py \
        --model_path "$MODEL_PATH" \
        --samples_per_category $SAMPLES_PER_CATEGORY \
        --max_new_tokens $MAX_NEW_TOKENS \
        --steps $STEPS \
        --step_interval $STEP_INTERVAL \
        --n_attribution_steps $N_ATTRIBUTION_STEPS \
        --output_dir "$OUTPUT_DIR" \
        --seed $SEED \
        --alg_temp 0.0 \
        --temperature 0.8 \
        --top_p 0.9 \
        --alg entropy \
        --device cuda
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Run $((i+1)) failed with exit code $EXIT_CODE"
        echo "Stopping execution."
        exit $EXIT_CODE
    fi
    
    echo ""
    echo "Run $((i+1))/3 completed at: $(date)"
    echo "=========================================="
    echo ""
done

echo ""
echo "=========================================="
echo "All 3 runs completed!"
echo "=========================================="
echo "Results saved to: $BASE_OUTPUT_DIR"
echo "  - Run 1: $BASE_OUTPUT_DIR/run_1_seed_42"
echo "  - Run 2: $BASE_OUTPUT_DIR/run_2_seed_123"
echo "  - Run 3: $BASE_OUTPUT_DIR/run_3_seed_2024"
echo ""
echo "To compare consistency across runs:"
echo "  python models/Dream/attribution/analyze_tools/compare_attribution_consistency.py --base_dir $BASE_OUTPUT_DIR"
echo ""
echo "To analyze attribution stability:"
echo "  python models/Dream/attribution/analyze_tools/analyze_attribution_stability.py --base_dir $BASE_OUTPUT_DIR"
echo "=========================================="

