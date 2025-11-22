#!/bin/bash
#SBATCH --job-name=llada_attribution_3runs
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
MODEL_PATH=${MODEL_PATH:-"GSAI-ML/LLaDA-8B-Base"}
SAMPLES_PER_CATEGORY=${SAMPLES_PER_CATEGORY:-10}
GEN_LENGTH=${GEN_LENGTH:-256}
STEPS=${STEPS:-256}
BLOCK_SIZE=${BLOCK_SIZE:-32}
N_ATTRIBUTION_STEPS=${N_ATTRIBUTION_STEPS:-10}
BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-"./attribution_results_$(date +%Y%m%d_%H%M%S)"}

# 指定使用GPU 4
export CUDA_VISIBLE_DEVICES=4

# 定义三个不同的随机种子
SEEDS=(47 123 2024)

echo "=========================================="
echo "LLaDA Head Attribution - 3 Runs Comparison"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Date: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL_PATH"
echo "Samples per category: $SAMPLES_PER_CATEGORY"
echo "Gen length: $GEN_LENGTH"
echo "Steps: $STEPS"
echo "Block size: $BLOCK_SIZE"
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
    
    python models/LLaDA/attribution/compute_nemotron_attribution.py \
        --model_path "$MODEL_PATH" \
        --samples_per_category $SAMPLES_PER_CATEGORY \
        --gen_length $GEN_LENGTH \
        --steps $STEPS \
        --block_size $BLOCK_SIZE \
        --n_attribution_steps $N_ATTRIBUTION_STEPS \
        --output_dir "$OUTPUT_DIR" \
        --seed $SEED \
        --device cuda
    
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
echo "  python models/LLaDA/attribution/compare_attribution_consistency.py --base_dir $BASE_OUTPUT_DIR"
echo "=========================================="

