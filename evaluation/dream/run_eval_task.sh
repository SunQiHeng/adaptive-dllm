#!/bin/bash
# Quick test script for Dream with standard, sparse, and adaptive modes
# Tests 3 model types on 2 tasks with reduced parameters
# Usage: bash run_eval_quick_test.sh

# Environment setup
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=/home/qiheng/Projects/adaptive-dllm:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=5

# Activate environment
# source ~/miniconda3/bin/activate adaptive-dllm

cd /home/qiheng/Projects/adaptive-dllm/evaluation/dream

# Create logs directory
mkdir -p logs results

# Model configuration (matching attribution script)
# NOTE:
# - `humaneval` is a *code completion* task (expects raw function body continuation).
#   Using chat template here makes the model emit explanations/markdown fences and tanks pass@1.
# - `humaneval_instruct` is designed for *instruct/chat* models and SHOULD use chat template.
MODEL_PATH="/data/qh_models/Dream-v0-Instruct-7B"
MODEL_TYPES=("sparse")

# Which precomputed head-importance to use for adaptive mode:
#
# Keep tag semantics aligned with LLaDA:
# - loss_gateIG:     use averaged loss-attribution scores
# - loss_gateIG_neg: use negated scores (ablation)
#
# Back-compat:
# - loss_gateIG_zero / loss_gateIG_zero_neg keep the older single-run paths.
IMPORTANCE_TAG=${IMPORTANCE_TAG:-"loss_gateIG_neg"}  # loss_gateIG | loss_gateIG_neg | loss_gateIG_zero | loss_gateIG_zero_neg | all_ones
SHUFFLE_SEED=${SHUFFLE_SEED:-1234}

if [ "$IMPORTANCE_TAG" = "loss_gateIG" ]; then
    PRECOMPUTED_IMPORTANCE_PATH="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream_loss_gateIG/head_importance.pt"
elif [ "$IMPORTANCE_TAG" = "loss_gateIG_neg" ]; then
    SRC_IMPORTANCE_PATH="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream_loss_gateIG/head_importance.pt"
    NEG_DIR="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream_loss_gateIG_neg"
    PRECOMPUTED_IMPORTANCE_PATH="${NEG_DIR}/head_importance.pt"
elif [ "$IMPORTANCE_TAG" = "loss_gateIG_zero" ]; then
    PRECOMPUTED_IMPORTANCE_PATH="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream_base_loss_gateIG_zero_maskp0.15-0.3-0.5-0.7-0.9_mcs2_mean_masked_seed47_n50_k8_L2048_dseed47_mseed47_ts20251227_191418/head_importance.pt"
elif [ "$IMPORTANCE_TAG" = "loss_gateIG_zero_neg" ]; then
    SRC_IMPORTANCE_PATH="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream_base_loss_gateIG_zero_maskp0.15-0.3-0.5-0.7-0.9_mcs2_mean_masked_seed47_n50_k8_L2048_dseed47_mseed47_ts20251227_191418/head_importance.pt"
    NEG_DIR="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream_base_loss_gateIG_zero_neg"
    PRECOMPUTED_IMPORTANCE_PATH="${NEG_DIR}/head_importance.pt"
elif [ "$IMPORTANCE_TAG" = "all_ones" ]; then
    PRECOMPUTED_IMPORTANCE_PATH="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream_base_all_ones/head_importance.pt"
else
    echo "ERROR: Unknown IMPORTANCE_TAG='$IMPORTANCE_TAG'. Use 'loss_gateIG', 'loss_gateIG_neg', 'loss_gateIG_zero', 'loss_gateIG_zero_neg', or 'all_ones'."
    exit 2
fi

# Auto-generate negated importance if needed
if [ "$IMPORTANCE_TAG" = "loss_gateIG_neg" ] || [ "$IMPORTANCE_TAG" = "loss_gateIG_zero_neg" ]; then
    if [ ! -f "$PRECOMPUTED_IMPORTANCE_PATH" ]; then
        echo "➖ Generating negated importance..."
        python /home/qiheng/Projects/adaptive-dllm/evaluation/dream/generate_negated_importance.py \
            --in_pt "$SRC_IMPORTANCE_PATH" \
            --out_dir "$NEG_DIR"
        if [ ! -f "$PRECOMPUTED_IMPORTANCE_PATH" ]; then
            echo "ERROR: Failed to generate negated importance at: $PRECOMPUTED_IMPORTANCE_PATH"
            exit 3
        fi
    else
        echo "➖ Using existing negated importance: $PRECOMPUTED_IMPORTANCE_PATH"
    fi
fi

echo "========================================================"
echo "Dream Quick Test Configuration"
echo "========================================================"
echo "Tasks: GSM8K, HumanEval"
echo "Model Types: standard, sparse, adaptive"
echo "Max New Tokens: 256"
echo "Block Size: 32"
echo "Test Samples: 50 per dataset"
echo "Importance tag: ${IMPORTANCE_TAG}"
echo "Importance file: ${PRECOMPUTED_IMPORTANCE_PATH}"
echo "========================================================"
echo ""

# Generation parameters (FIXED to match official Dream eval)
# CRITICAL: Official Dream uses temperature=0.1 and alg_temp=0.0, NOT 0.8/1.5!
TEMPERATURE=0.1  # Official: 0.1 (NOT 0.8!)
TOP_P=0.9
ALG="entropy"
ALG_TEMP=0.0  # Official: 0.0 (NOT 1.5!)
BLOCK_SIZE=32
LIMIT=150

# Task-specific parameters (will be set per task)
MAX_NEW_TOKENS=256
STEPS=256

# Sparse parameters
SKIP=0.2
SELECT=0.3


# Tasks to run
TASKS=("gsm8k" "humaneval")

# Function to run evaluation for one model type on one task
run_single_eval() {
    local task=$1
    local model_type=$2
    
    echo ""
    echo "========================================"
    echo "Running: ${model_type} on ${task}"
    echo "========================================"
    
    OUTPUT_DIR="results/${model_type}/${task}_${IMPORTANCE_TAG}"
    mkdir -p "$OUTPUT_DIR"
    
    # Record start time
    START_TIME=$(date +%s)
    
    # Set task-specific parameters (matching official Dream eval)
    if [ "$task" = "humaneval" ] || [ "$task" = "humaneval_instruct" ]; then
        MAX_NEW_TOKENS=768
        STEPS=768
    elif [ "$task" = "gsm8k" ] || [ "$task" = "gsm8k_cot" ]; then
        MAX_NEW_TOKENS=256
        STEPS=256
    else
        MAX_NEW_TOKENS=256
        STEPS=256
    fi
    
    echo "Params: max_new_tokens=${MAX_NEW_TOKENS}, steps=${STEPS}, temperature=${TEMPERATURE}, alg_temp=${ALG_TEMP}, block_size=${BLOCK_SIZE}, limit=${LIMIT}"
    
    # Set importance source for adaptive mode
    if [ "$model_type" = "adaptive" ]; then
        # GQA weighting granularity for Dream adaptive attention:
        # - kv: average weights within each KV group (often more stable for GQA; recommended default for Dream)
        # - q : apply weights per query head (preserves attribution resolution, but can be noisier under GQA)
        GQA_WEIGHT_MODE=${GQA_WEIGHT_MODE:-"kv"}
        # How strong the adaptive reallocation is. Smaller => closer to uniform sparse (often safer).
        RELATIVE_WEIGHT_SCALE=${RELATIVE_WEIGHT_SCALE:-"0.6666667"}
        # Safety clamp to avoid empty masks for very low-weight heads.
        MIN_KEEP_RATIO=${MIN_KEEP_RATIO:-"0.1"}
        IMPORTANCE_ARG=",importance_source=precomputed,precomputed_importance_path=${PRECOMPUTED_IMPORTANCE_PATH},gqa_weight_mode=${GQA_WEIGHT_MODE},relative_weight_scale=${RELATIVE_WEIGHT_SCALE},min_keep_ratio=${MIN_KEEP_RATIO}"
    else
        IMPORTANCE_ARG=""
    fi
    
    # Build a concrete command (and persist it) so we can diff runs reliably.
    # NOTE: lm-eval logs `Initializing dream_eval model, with arguments: {...}` which is the *authoritative*
    # view of what reached DreamEvalHarness, but saving the full CLI helps catch shell/env differences.
    # IMPORTANT: Do NOT embed extra quotes inside model_args values.
    # lm-eval's arg-string parser does not strip them, so model_type would become '\"sparse\"' and fail.
    # Shell quoting around the whole --model_args string is sufficient.
    MODEL_ARGS_STR="model_path=${MODEL_PATH},model_type=${model_type},max_new_tokens=${MAX_NEW_TOKENS},steps=${STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},alg=${ALG},alg_temp=${ALG_TEMP},skip=${SKIP},select=${SELECT},block_size=${BLOCK_SIZE}${IMPORTANCE_ARG}"

    # Record environment signature alongside the command (helps diagnose version-induced drift).
    {
        echo "[env] date: $(date -Iseconds)"
        echo "[env] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"(unset)"}"
        python - <<'PY'
import platform
try:
    import torch
except Exception as e:
    torch = None
    print("[env] torch import failed:", e)
try:
    import transformers
except Exception as e:
    transformers = None
    print("[env] transformers import failed:", e)
print("[env] python:", platform.python_version())
print("[env] torch:", getattr(torch, "__version__", None))
print("[env] transformers:", getattr(transformers, "__version__", None))
PY
    } > "${OUTPUT_DIR}/run_env.txt" 2>&1

    # Build the command based on task
    if [ "$task" = "humaneval" ]; then
        # HumanEval (non-instruct) expects raw code completion; DO NOT apply chat template.
        # HumanEval requires --confirm_run_unsafe_code
        CMD=(python -m accelerate.commands.launch --num_processes=1 eval_dream.py
            --model dream_eval
            --model_args "${MODEL_ARGS_STR}"
            --tasks "${task}"
            --num_fewshot 0
            --limit "${LIMIT}"
            --output_path "${OUTPUT_DIR}/results.json"
            --log_samples
            --confirm_run_unsafe_code
        )
    elif [ "$task" = "humaneval_instruct" ]; then
        # HumanEval-Instruct is designed for chat/instruct models; apply chat template.
        CMD=(python -m accelerate.commands.launch --num_processes=1 eval_dream.py
            --model dream_eval
            --model_args "${MODEL_ARGS_STR}"
            --tasks "${task}"
            --num_fewshot 0
            --limit "${LIMIT}"
            --output_path "${OUTPUT_DIR}/results.json"
            --log_samples
            --apply_chat_template
            --confirm_run_unsafe_code
        )
    else
        # GSM8K and other generation tasks
        CMD=(python -m accelerate.commands.launch --num_processes=1 eval_dream.py
            --model dream_eval
            --model_args "${MODEL_ARGS_STR}"
            --tasks "${task}"
            --num_fewshot 0
            --limit "${LIMIT}"
            --output_path "${OUTPUT_DIR}/results.json"
            --log_samples
            --apply_chat_template
        )
    fi

    # Persist the exact command in a copy/paste-able form.
    # `printf %q` emits a safely shell-escaped command line.
    {
        printf "%q " "${CMD[@]}"
        echo
    } > "${OUTPUT_DIR}/run_cmd.sh"
    echo "[run] saved command to ${OUTPUT_DIR}/run_cmd.sh"
    echo "[run] saved env to ${OUTPUT_DIR}/run_env.txt"

    # Execute
    "${CMD[@]}" 2>&1 | tee "${OUTPUT_DIR}/eval.log"
    
    # Calculate running time
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))
    ELAPSED_SEC=$((ELAPSED % 60))
    
    # Record time to file
    echo "${ELAPSED}" > "${OUTPUT_DIR}/runtime.txt"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ${model_type} - ${task} - ${ELAPSED}s (${ELAPSED_MIN}m ${ELAPSED_SEC}s)" >> "results/timing_log.txt"
    
    echo "✅ Completed ${model_type} on ${task}"
    echo "⏱️  Running time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s (${ELAPSED}s total)"
    echo ""
}

# Main execution
echo "🚀 Starting Dream quick test evaluation..."
echo "Started at: $(date)"
echo ""

# Total tasks counter
TOTAL_TASKS=$((${#TASKS[@]} * ${#MODEL_TYPES[@]}))
CURRENT_TASK=0

# Run all combinations
for task in "${TASKS[@]}"; do
    echo ""
    echo "================================================"
    echo "📊 Task: ${task^^}"
    echo "================================================"
    
    for model_type in "${MODEL_TYPES[@]}"; do
        CURRENT_TASK=$((CURRENT_TASK + 1))
        echo ""
        echo "Progress: [${CURRENT_TASK}/${TOTAL_TASKS}]"
        
        run_single_eval "$task" "$model_type"
    done
done

echo ""
echo "================================================"
echo "✨ All evaluations completed!"
echo "Finished at: $(date)"
echo "================================================"
echo ""
echo "📁 Results saved in: results/"
echo "📊 Timing log: results/timing_log.txt"
echo ""

# Generate a summary
echo "📈 Summary:"
echo ""
for task in "${TASKS[@]}"; do
    echo "Task: ${task}"
    for model_type in "${MODEL_TYPES[@]}"; do
        RESULT_FILE="results/${model_type}/${task}_${IMPORTANCE_TAG}/results.json"
        if [ -f "$RESULT_FILE" ]; then
            echo "  ✅ ${model_type}: results/${model_type}/${task}_${IMPORTANCE_TAG}/"
        else
            echo "  ❌ ${model_type}: FAILED"
        fi
    done
    echo ""
done

