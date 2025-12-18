#!/bin/bash
# Direct run on GPU (can run in background with nohup)

# Navigate to project root first
cd /home/qiheng/Projects/adaptive-dllm/models/Dream/attribution

# Create logs directory
mkdir -p logs

# Configuration
MODEL_PATH=${MODEL_PATH:-"/data/qh_models/Dream-v0-Instruct-7B"}
SAMPLES_PER_CATEGORY=${SAMPLES_PER_CATEGORY:-20}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
STEPS=${STEPS:-32}
STEP_INTERVAL=${STEP_INTERVAL:-8}
N_ATTRIBUTION_STEPS=${N_ATTRIBUTION_STEPS:-10}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-"/home/qiheng/Projects/adaptive-dllm/models/Dream/attribution/attribution_results_${TIMESTAMP}"}

# 指定使用GPU (默认GPU 2，可通过环境变量修改)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3}

# 定义三个不同的随机种子
SEEDS=(42 123 2024)

LOGFILE="logs/attribution_direct_${TIMESTAMP}.log"
PIDFILE="logs/attribution_direct_${TIMESTAMP}.pid"

# 保存进程ID
echo $$ > "$PIDFILE"

echo "=========================================="
echo "Dream Head Attribution - 3 Runs Comparison"
echo "=========================================="
echo "Date: $(date)"
echo "PID: $$"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL_PATH"
echo "Samples per category: $SAMPLES_PER_CATEGORY"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Diffusion steps: $STEPS"
echo "Step interval: $STEP_INTERVAL"
echo "Attribution steps: $N_ATTRIBUTION_STEPS"
echo "Base output dir: $BASE_OUTPUT_DIR"
echo "Seeds: ${SEEDS[@]}"
echo "Log file: $LOGFILE"
echo "PID file: $PIDFILE"
echo "=========================================="

# Navigate to project root
cd /home/qiheng/Projects/adaptive-dllm

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
echo ""
echo "Job finished at: $(date)"

# Clean up PID file
rm -f "$PIDFILE"

