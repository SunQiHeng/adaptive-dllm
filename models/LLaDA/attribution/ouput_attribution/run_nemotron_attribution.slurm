#!/bin/bash
# LLaDA Head Attribution - Can run with SLURM or directly
# 
# SLURM usage: sbatch run_nemotron_attribution.slurm
# Direct usage: bash run_nemotron_attribution.slurm
# Background:   nohup bash run_nemotron_attribution.slurm > logs/run.log 2>&1 &

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

# Navigate to attribution directory
cd /home/qiheng/Projects/adaptive-dllm/models/LLaDA/attribution

# Create logs directory
mkdir -p logs

# Configuration
MODEL_PATH=${MODEL_PATH:-"GSAI-ML/LLaDA-8B-Base"}
SAMPLES_PER_CATEGORY=${SAMPLES_PER_CATEGORY:-10}
GEN_LENGTH=${GEN_LENGTH:-256}
STEPS=${STEPS:-256}
BLOCK_SIZE=${BLOCK_SIZE:-32}
N_ATTRIBUTION_STEPS=${N_ATTRIBUTION_STEPS:-10}
# Attribution objective config
# Supported: target_logprob | target_logit | margin
OBJECTIVE=${OBJECTIVE:-"target_logit"}
# Only used when OBJECTIVE=margin. 1 to use logprob margin, 0 to use logit margin.
MARGIN_USE_LOGPROB=${MARGIN_USE_LOGPROB:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# NOTE:
# - In SLURM jobs, the scheduler typically sets CUDA_VISIBLE_DEVICES for you.
#   Overriding it here can accidentally select the wrong GPU or a non-existent one.
# - In direct runs (no SLURM), you may set CUDA_VISIBLE_DEVICES manually; otherwise we pick a safe default.

# 定义三个不同的随机种子
SEEDS=(47 123 2024)

# 检测是否在 SLURM 环境中运行
if [ -n "$SLURM_JOB_ID" ]; then
    RUNNING_MODE="SLURM"
    JOB_INFO="Job ID: $SLURM_JOB_ID, Node: $SLURM_NODELIST"
    LOGFILE="logs/attribution_3runs_${SLURM_JOB_ID}.out"
    DEFAULT_BASE_OUTPUT_DIR="/home/qiheng/Projects/adaptive-dllm/models/LLaDA/attribution/attribution_results_${OBJECTIVE}_${TIMESTAMP}_job${SLURM_JOB_ID}"
else
    RUNNING_MODE="Direct"
    JOB_INFO="PID: $$"
    LOGFILE="logs/attribution_direct_${TIMESTAMP}.log"
    PIDFILE="logs/attribution_direct_${TIMESTAMP}.pid"
    # 保存进程ID（直接运行时）
    echo $$ > "$PIDFILE"
    DEFAULT_BASE_OUTPUT_DIR="/home/qiheng/Projects/adaptive-dllm/models/LLaDA/attribution/attribution_results_${OBJECTIVE}_${TIMESTAMP}"
    # 指定使用 GPU（direct 模式下的默认值，可通过环境变量修改）
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
fi

# Base output directory (can be overridden via env var)
BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-"$DEFAULT_BASE_OUTPUT_DIR"}

echo "=========================================="
echo "LLaDA Head Attribution - 3 Runs Comparison"
echo "=========================================="
echo "Running mode: $RUNNING_MODE"
echo "$JOB_INFO"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "Model: $MODEL_PATH"
echo "Samples per category: $SAMPLES_PER_CATEGORY"
echo "Gen length: $GEN_LENGTH"
echo "Steps: $STEPS"
echo "Block size: $BLOCK_SIZE"
echo "Attribution steps: $N_ATTRIBUTION_STEPS"
echo "Objective: $OBJECTIVE (margin_use_logprob=$MARGIN_USE_LOGPROB)"
echo "Base output dir: $BASE_OUTPUT_DIR"
echo "Seeds: ${SEEDS[@]}"
if [ "$RUNNING_MODE" = "Direct" ]; then
    echo "Log file: $LOGFILE"
    echo "PID file: $PIDFILE"
fi
echo "=========================================="

# Navigate to project root
cd /home/qiheng/Projects/adaptive-dllm

# Run attribution 3 times with different seeds
for i in {0..2}; do
    SEED=${SEEDS[$i]}
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/run_$((i+1))_seed_${SEED}"

    EXTRA_ARGS=()
    if [ "$OBJECTIVE" = "margin" ] && [ "$MARGIN_USE_LOGPROB" != "0" ]; then
        EXTRA_ARGS+=(--objective_margin_use_logprob)
    fi
    
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
        --objective "$OBJECTIVE" \
        "${EXTRA_ARGS[@]}" \
        --output_dir "$OUTPUT_DIR" \
        --seed $SEED \
        --device cuda
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Run $((i+1)) failed with exit code $EXIT_CODE"
        echo "Stopping execution."
        # 清理 PID 文件（如果存在）
        [ -f "$PIDFILE" ] && rm -f "$PIDFILE"
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
echo "  - Run 1: $BASE_OUTPUT_DIR/run_1_seed_47"
echo "  - Run 2: $BASE_OUTPUT_DIR/run_2_seed_123"
echo "  - Run 3: $BASE_OUTPUT_DIR/run_3_seed_2024"
echo ""
echo "To compare consistency across runs:"
echo "  python models/LLaDA/attribution/compare_attribution_consistency.py --base_dir $BASE_OUTPUT_DIR"
echo "=========================================="
echo ""
echo "Job finished at: $(date)"

# 清理 PID 文件（直接运行时）
[ -f "$PIDFILE" ] && rm -f "$PIDFILE"

