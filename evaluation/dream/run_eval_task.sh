#!/bin/bash
# Quick test script for Dream with standard, sparse, and adaptive modes
# Tests 3 model types on 2 tasks with reduced parameters
# Usage: bash run_eval_quick_test.sh

# Make pipelines fail if the left-hand command fails (e.g., when piping to tee).
set -o pipefail

# Project root (auto-detected, but can be overridden)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-"$(cd "${SCRIPT_DIR}/../.." && pwd)"}"

# Environment setup
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH="${PROJECT_ROOT}:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=1

# Activate environment
# source ~/miniconda3/bin/activate adaptive-dllm

cd "${PROJECT_ROOT}/evaluation/dream"

# Create logs directory
mkdir -p logs results

# Model configuration (matching attribution script)
# NOTE:
# - `humaneval` is a *code completion* task (expects raw function body continuation).
#   Using chat template here makes the model emit explanations/markdown fences and tanks pass@1.
# - `humaneval_instruct` is designed for *instruct/chat* models and SHOULD use chat template.
MODEL_PATH="/data/qh_models/Dream-v0-Instruct-7B"
MODEL_TYPES=("adaptive")

# -------------------------
# Importance path selection
# -------------------------
# You can manually choose the score file by setting:
#   PRECOMPUTED_IMPORTANCE_PATH=/path/to/head_importance.pt bash run_eval_task.sh
#
# This script will then set:
#   RECOMPUTED_IMPORTANCE_PATH=${PRECOMPUTED_IMPORTANCE_PATH:-"<default>"}
# and (optionally) negate it based on USE_NEGATED.
#
# Whether to negate the scores (0=use original, 1=negate)
USE_NEGATED=${USE_NEGATED:-1}

# Default base importance (if not provided via env)
DEFAULT_IMPORTANCE_PATH="${PROJECT_ROOT}/configs/head_importance_dream_loss_gateIG/head_importance.pt"

# Base score path (user-selectable)
PRECOMPUTED_IMPORTANCE_PATH=${PRECOMPUTED_IMPORTANCE_PATH:-"${DEFAULT_IMPORTANCE_PATH}"}

# Final score path (what we actually pass downstream)
RECOMPUTED_IMPORTANCE_PATH=${RECOMPUTED_IMPORTANCE_PATH:-"${PRECOMPUTED_IMPORTANCE_PATH}"}

# Auto-generate negated importance if requested
if [ "${USE_NEGATED}" = "1" ]; then
    SRC_IMPORTANCE_PATH="${RECOMPUTED_IMPORTANCE_PATH}"
    NEG_DIR=${NEG_DIR:-"$(dirname "${SRC_IMPORTANCE_PATH}")_neg"}
    RECOMPUTED_IMPORTANCE_PATH="${NEG_DIR}/head_importance.pt"
    if [ ! -f "${RECOMPUTED_IMPORTANCE_PATH}" ]; then
        echo "➖ Generating negated importance..."
        python "${SCRIPT_DIR}/generate_negated_importance.py" \
            --in_pt "${SRC_IMPORTANCE_PATH}" \
            --out_dir "${NEG_DIR}"
        if [ ! -f "${RECOMPUTED_IMPORTANCE_PATH}" ]; then
            echo "ERROR: Failed to generate negated importance at: ${RECOMPUTED_IMPORTANCE_PATH}"
            exit 3
        fi
    else
        echo "➖ Using existing negated importance: ${RECOMPUTED_IMPORTANCE_PATH}"
    fi
fi

# Tag for output directory naming only (can be overridden)
IMPORTANCE_TAG=${IMPORTANCE_TAG:-"manual$( [ "${USE_NEGATED}" = "1" ] && echo "_neg" )"}

echo "========================================================"
echo "Dream Quick Test Configuration"
echo "========================================================"
echo "Tasks: ${TASKS[*]}"
echo "Model Types: standard, sparse, adaptive"
echo "Max New Tokens: 256"
echo "Block Size: 32"
echo "Test Samples: 50 per dataset"
echo "Importance tag: ${IMPORTANCE_TAG}"
echo "Importance base: ${PRECOMPUTED_IMPORTANCE_PATH}"
echo "Importance used: ${RECOMPUTED_IMPORTANCE_PATH}"
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


# Tasks to run (can be overridden without editing file):
#   TASKS_STR="mmlu" bash run_eval_task.sh
TASKS=("gsm8k" "humaneval")
if [ -n "${TASKS_STR:-}" ]; then
    IFS=',' read -r -a TASKS <<< "${TASKS_STR}"
fi

# Function to run evaluation for one model type on one task
run_single_eval() {
    local task=$1
    local model_type=$2
    
    echo ""
    echo "========================================"
    echo "Running: ${model_type} on ${task}"
    echo "========================================"
    
    # Task tag for output directory (avoid overwriting when toggling chat template, etc.)
    TASK_TAG="${task}"
    if [ "${task}" = "mmlu" ]; then
        TASK_TAG="mmlu_chat"
    fi

    OUTPUT_DIR="results/${model_type}/${TASK_TAG}_${IMPORTANCE_TAG}"
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
        IMPORTANCE_ARG=",importance_source=precomputed,precomputed_importance_path=${RECOMPUTED_IMPORTANCE_PATH},gqa_weight_mode=${GQA_WEIGHT_MODE},relative_weight_scale=${RELATIVE_WEIGHT_SCALE},min_keep_ratio=${MIN_KEEP_RATIO}"
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
    elif [ "$task" = "mmlu" ]; then
        # MMLU is a multiple-choice likelihood task. Follow eval_dream.sh:
        # - use few-shot (default 5)
        # - batch_size=1 (passed via lm-eval CLI, NOT via --model_args to avoid duplicate kwarg)
        # - Apply chat template by default to align with attribution context
        # - For sparse/adaptive: enable likelihood_now_step and recompute_mask_each_call to trigger sparse attention
        MMLU_FEWSHOT=${MMLU_FEWSHOT:-5}
        CMD=(python -m accelerate.commands.launch --num_processes=1 eval_dream.py
            --model dream_eval
            --model_args "${MODEL_ARGS_STR},mc_num=1,likelihood_now_step=${STEPS},recompute_mask_each_call=true"
            --tasks "${task}"
            --num_fewshot "${MMLU_FEWSHOT}"
            --batch_size 1
            --limit "${LIMIT}"
            --output_path "${OUTPUT_DIR}/results.json"
            --log_samples
            --apply_chat_template
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
    CMD_RC=${PIPESTATUS[0]}
    
    # Calculate running time
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))
    ELAPSED_SEC=$((ELAPSED % 60))
    
    # Record time to file
    echo "${ELAPSED}" > "${OUTPUT_DIR}/runtime.txt"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ${model_type} - ${task} - ${ELAPSED}s (${ELAPSED_MIN}m ${ELAPSED_SEC}s)" >> "results/timing_log.txt"
    
    if [ "${CMD_RC}" -eq 0 ]; then
        echo "✅ Completed ${model_type} on ${task}"
    else
        echo "❌ Failed ${model_type} on ${task} (exit=${CMD_RC})"
        echo "   See: ${OUTPUT_DIR}/eval.log"
    fi
    echo "⏱️  Running time: ${ELAPSED_MIN}m ${ELAPSED_SEC}s (${ELAPSED}s total)"
    echo ""

    return "${CMD_RC}"
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
        
        run_single_eval "$task" "$model_type" || exit $?
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

