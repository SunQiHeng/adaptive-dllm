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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

# Activate environment
# source ~/miniconda3/bin/activate adaptive-dllm

cd "${PROJECT_ROOT}/evaluation/dream"

# Create logs directory
mkdir -p logs results

# Model configuration (matching attribution script)
# NOTE:
# - `humaneval` is a code-completion task (expects raw function body continuation).
# - `mbpp` is also treated as a code task and uses the unsafe-code evaluator.
# - `humaneval_instruct` is designed for instruct/chat models and SHOULD use chat template.
MODEL_PATH=${MODEL_PATH:-"/data/qh_models/Dream-v0-Instruct-7B"}
MODEL_TYPES=("adaptive")
if [ -n "${MODEL_TYPES_STR:-}" ]; then
    IFS=',' read -r -a MODEL_TYPES <<< "${MODEL_TYPES_STR}"
fi

# -------------------------
# Importance score path selection (EDIT ONE LINE)
# -------------------------
# Only change the next line (or override via env IMPORTANCE_PATH=...):
IMPORTANCE_PATH=${IMPORTANCE_PATH:-"${PROJECT_ROOT}/configs/head_importance_dream_mmlu_all_pmrandom_threshold_ts20260323_224941/head_importance.pt"}
TASKS=("mmlu")
if [ -n "${TASKS_STR:-}" ]; then
    IFS=',' read -r -a TASKS <<< "${TASKS_STR}"
fi
LIMIT=${LIMIT:-10}
GPQA_DATA_PATH=${GPQA_DATA_PATH:-""}
LOCAL_TASK_ROOT="${PROJECT_ROOT}/evaluation/local_tasks/generated"
DEFAULT_GPQA_DATA_PATH="${PROJECT_ROOT}/evaluation/local_data/gpqa/gpqa_main.jsonl"
if [ -z "${GPQA_DATA_PATH}" ] && [ -f "${DEFAULT_GPQA_DATA_PATH}" ]; then
    GPQA_DATA_PATH="${DEFAULT_GPQA_DATA_PATH}"
fi

check_minerva_math_deps() {
    python - <<'PY'
import importlib
missing = []
for mod in ("antlr4", "sympy", "math_verify"):
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)
if missing:
    print("ERROR: minerva_math requires additional math evaluation dependencies.")
    print("Missing modules:", ", ".join(missing))
    print("Install with: pip install antlr4-python3-runtime==4.11 sympy math_verify")
    raise SystemExit(2)
PY
}

check_gpqa_access() {
    python - <<'PY'
from datasets import load_dataset_builder
try:
    load_dataset_builder("Idavidrein/gpqa", "gpqa_main")
except Exception as e:
    msg = str(e)
    if "gated" in msg.lower() or "access" in msg.lower():
        print("ERROR: gpqa_main_n_shot requires access to the gated HF dataset 'Idavidrein/gpqa'.")
        print("Request access on Hugging Face or switch to a locally exported dataset/task implementation.")
        raise SystemExit(2)
    raise
PY
}

prepare_gpqa_local_task() {
    if [ -z "${GPQA_DATA_PATH}" ]; then
        return 1
    fi
    local gpqa_task_dir="${LOCAL_TASK_ROOT}/gpqa_local"
    mkdir -p "${LOCAL_TASK_ROOT}"
    python "${PROJECT_ROOT}/evaluation/local_tasks/prepare_gpqa_local_task.py" \
        --data_path "${GPQA_DATA_PATH}" \
        --output_dir "${gpqa_task_dir}"
}

get_gpqa_local_row_count() {
    python - <<'PY'
import json
from pathlib import Path
import os

data_path = Path(os.environ["GPQA_DATA_PATH"])
count = 0
files = [data_path] if data_path.is_file() else sorted(
    p for p in data_path.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".jsonl"}
)
for path in files:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
    else:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            count += len(obj)
        elif isinstance(obj, dict):
            for key in ("train", "validation", "test", "rows", "data", "examples"):
                if isinstance(obj.get(key), list):
                    count += len(obj[key])
                    break
            else:
                count += 1
print(count)
PY
}
# Whether to negate the scores (0=use original, 1=negate). Keep default aligned with prior behavior.
USE_NEGATED=${USE_NEGATED:-1}

# What we actually pass downstream
USED_IMPORTANCE_PATH="${IMPORTANCE_PATH}"

# Auto-generate negated importance if requested
if [ "${USE_NEGATED}" = "1" ]; then
    SRC_IMPORTANCE_PATH="${IMPORTANCE_PATH}"
    NEG_DIR=${NEG_DIR:-"$(dirname "${SRC_IMPORTANCE_PATH}")_neg"}
    USED_IMPORTANCE_PATH="${NEG_DIR}/head_importance.pt"
    if [ ! -f "${USED_IMPORTANCE_PATH}" ]; then
        echo "➖ Generating negated importance..."
        python "${SCRIPT_DIR}/generate_negated_importance.py" \
            --in_pt "${SRC_IMPORTANCE_PATH}" \
            --out_dir "${NEG_DIR}"
        if [ ! -f "${USED_IMPORTANCE_PATH}" ]; then
            echo "ERROR: Failed to generate negated importance at: ${USED_IMPORTANCE_PATH}"
            exit 3
        fi
    else
        echo "➖ Using existing negated importance: ${USED_IMPORTANCE_PATH}"
    fi
fi

# Tag for output directory naming only (can be overridden)
DEFAULT_IMPORTANCE_TAG="$(basename "$(dirname "${IMPORTANCE_PATH}")")$( [ "${USE_NEGATED}" = "1" ] && echo "_neg" )"
IMPORTANCE_TAG=${IMPORTANCE_TAG:-"${DEFAULT_IMPORTANCE_TAG}"}

echo "========================================================"
echo "Dream Quick Test Configuration"
echo "========================================================"
echo "Tasks: ${TASKS[*]}"
echo "Model Types: ${MODEL_TYPES[*]}"
echo "Default Max New Tokens/Steps: 256/256 (task overrides apply for humaneval, mbpp, and minerva_math)"
echo "Block Size: 32"
echo "Limit: ${LIMIT}"
echo "Importance tag: ${IMPORTANCE_TAG}"
echo "Importance base: ${IMPORTANCE_PATH}"
echo "Importance used: ${USED_IMPORTANCE_PATH}"
echo "========================================================"
echo ""

# Generation parameters (FIXED to match official Dream eval)
# CRITICAL: Official Dream uses temperature=0.1 and alg_temp=0.0, NOT 0.8/1.5!
TEMPERATURE=0.1  # Official: 0.1 (NOT 0.8!)
TOP_P=0.9
ALG="entropy"
ALG_TEMP=0.0  # Official: 0.0 (NOT 1.5!)
BLOCK_SIZE=32
BLOCK_LENGTH=${BLOCK_LENGTH:-32}
DIFFUSION_MODE=${DIFFUSION_MODE:-"global"}


# Task-specific parameters (will be set per task)
MAX_NEW_TOKENS=256
STEPS=256

# Sparse parameters
SKIP=0.2
SELECT=0.3


# Tasks to run (can be overridden without editing file):
#   TASKS_STR="mmlu,cmmlu,ceval-valid,gsm8k,minerva_math,gpqa_main_n_shot,humaneval,mbpp" bash run_eval_task.sh

# Function to run evaluation for one model type on one task
run_single_eval() {
    local task=$1
    local model_type=$2
    local task_name="${task}"
    
    echo ""
    echo "========================================"
    echo "Running: ${model_type} on ${task}"
    echo "========================================"
    
    # Task tag for output directory (avoid overwriting when toggling chat template, etc.)
    TASK_TAG="${task}"
    case "${task}" in
        mmlu|cmmlu|ceval-valid|gpqa_main_n_shot)
            TASK_TAG="${task}_chat"
            ;;
    esac

    OUTPUT_DIR="results/${model_type}/${TASK_TAG}_${IMPORTANCE_TAG}"
    mkdir -p "$OUTPUT_DIR"
    local progress_label="${task}|${model_type}|${TASK_TAG}_${IMPORTANCE_TAG}"
    
    # Record start time
    START_TIME=$(date +%s)
    
    # Set task-specific parameters (matching official Dream eval)
    if [ "$task" = "humaneval" ] || [ "$task" = "humaneval_instruct" ]; then
        MAX_NEW_TOKENS=768
        STEPS=768
    elif [ "$task" = "mbpp" ]; then
        MAX_NEW_TOKENS=${MBPP_MAX_NEW_TOKENS:-512}
        STEPS=${MBPP_STEPS:-${MAX_NEW_TOKENS}}
    elif [ "$task" = "gsm8k" ] || [ "$task" = "gsm8k_cot" ]; then
        MAX_NEW_TOKENS=256
        STEPS=256
    elif [ "$task" = "minerva_math" ]; then
        MAX_NEW_TOKENS=${MINERVA_MATH_MAX_NEW_TOKENS:-512}
        STEPS=${MINERVA_MATH_STEPS:-${MAX_NEW_TOKENS}}
    else
        MAX_NEW_TOKENS=256
        STEPS=256
    fi

    case "$task" in
        minerva_math)
            check_minerva_math_deps || return $?
            ;;
        gpqa_main_n_shot)
            if [ -z "${GPQA_DATA_PATH}" ]; then
                check_gpqa_access || return $?
            fi
            ;;
    esac
    
    echo "Params: max_new_tokens=${MAX_NEW_TOKENS}, steps=${STEPS}, temperature=${TEMPERATURE}, alg_temp=${ALG_TEMP}, block_size=${BLOCK_SIZE}, block_length=${BLOCK_LENGTH}, diffusion_mode=${DIFFUSION_MODE}, limit=${LIMIT}"
    
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
        IMPORTANCE_ARG=",importance_source=precomputed,precomputed_importance_path=${USED_IMPORTANCE_PATH},gqa_weight_mode=${GQA_WEIGHT_MODE},relative_weight_scale=${RELATIVE_WEIGHT_SCALE},min_keep_ratio=${MIN_KEEP_RATIO}"
    else
        IMPORTANCE_ARG=""
    fi
    INCLUDE_PATH_ARGS=()
    if [ "$task" = "gpqa_main_n_shot" ] && [ -n "${GPQA_DATA_PATH}" ]; then
        GPQA_TASK_DIR="$(prepare_gpqa_local_task)" || return $?
        INCLUDE_PATH_ARGS=(--include_path "${GPQA_TASK_DIR}")
        task_name="gpqa_main_n_shot_local"
        GPQA_LOCAL_ROWS="$(get_gpqa_local_row_count)" || return $?
    fi
    
    # Build a concrete command (and persist it) so we can diff runs reliably.
    # NOTE: lm-eval logs `Initializing dream_eval model, with arguments: {...}` which is the *authoritative*
    # view of what reached DreamEvalHarness, but saving the full CLI helps catch shell/env differences.
    # IMPORTANT: Do NOT embed extra quotes inside model_args values.
    # lm-eval's arg-string parser does not strip them, so model_type would become '\"sparse\"' and fail.
    # Shell quoting around the whole --model_args string is sufficient.
    MODEL_ARGS_STR="model_path=${MODEL_PATH},model_type=${model_type},max_new_tokens=${MAX_NEW_TOKENS},steps=${STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},alg=${ALG},alg_temp=${ALG_TEMP},diffusion_mode=${DIFFUSION_MODE},block_length=${BLOCK_LENGTH},skip=${SKIP},select=${SELECT},block_size=${BLOCK_SIZE},progress_label=${progress_label}${IMPORTANCE_ARG}"

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
    if [ "$task" = "humaneval" ] || [ "$task" = "mbpp" ]; then
        # HumanEval (non-instruct) expects raw code completion; DO NOT apply chat template.
        # MBPP also uses code execution-based evaluation.
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
    elif [ "$task" = "mmlu" ] || [ "$task" = "cmmlu" ] || [ "$task" = "ceval-valid" ] || [ "$task" = "gpqa_main_n_shot" ]; then
        # Multiple-choice likelihood tasks share the same fast sparse/adaptive-compatible setup.
        # Use 5-shot by default to match the official scripts for MMLU-family tasks and GPQA.
        if [ "$task" = "gpqa_main_n_shot" ]; then
            NUM_FEWSHOT_LOCAL=${GPQA_FEWSHOT:-5}
            if [ -n "${GPQA_DATA_PATH}" ]; then
                GPQA_LOCAL_MAX_FEWSHOT=$(( GPQA_LOCAL_ROWS > 1 ? GPQA_LOCAL_ROWS - 1 : 0 ))
                if [ "${NUM_FEWSHOT_LOCAL}" -gt "${GPQA_LOCAL_MAX_FEWSHOT}" ]; then
                    echo "[gpqa_local] Requested num_fewshot=${NUM_FEWSHOT_LOCAL}, but local dataset has only ${GPQA_LOCAL_ROWS} rows. Clamping to ${GPQA_LOCAL_MAX_FEWSHOT}."
                    NUM_FEWSHOT_LOCAL=${GPQA_LOCAL_MAX_FEWSHOT}
                fi
            fi
        else
            NUM_FEWSHOT_LOCAL=${MMLU_FEWSHOT:-5}
        fi
        CMD=(python -m accelerate.commands.launch --num_processes=1 eval_dream.py
            --model dream_eval
            --model_args "${MODEL_ARGS_STR},mc_num=1,likelihood_now_step=${STEPS},recompute_mask_each_call=true"
            --tasks "${task_name}"
            "${INCLUDE_PATH_ARGS[@]}"
            --num_fewshot "${NUM_FEWSHOT_LOCAL}"
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
            --tasks "${task_name}"
            "${INCLUDE_PATH_ARGS[@]}"
            --num_fewshot 0
            --limit "${LIMIT}"
            --output_path "${OUTPUT_DIR}/results.json"
            --log_samples
            --apply_chat_template
            --confirm_run_unsafe_code
        )
    elif [ "$task" = "gsm8k" ] || [ "$task" = "gsm8k_cot" ] || [ "$task" = "minerva_math" ]; then
        CMD=(python -m accelerate.commands.launch --num_processes=1 eval_dream.py
            --model dream_eval
            --model_args "${MODEL_ARGS_STR}"
            --tasks "${task_name}"
            "${INCLUDE_PATH_ARGS[@]}"
            --num_fewshot 0
            --limit "${LIMIT}"
            --output_path "${OUTPUT_DIR}/results.json"
            --log_samples
            --apply_chat_template
        )
    else
        CMD=(python -m accelerate.commands.launch --num_processes=1 eval_dream.py
            --model dream_eval
            --model_args "${MODEL_ARGS_STR}"
            --tasks "${task_name}"
            "${INCLUDE_PATH_ARGS[@]}"
            --num_fewshot 0
            --limit "${LIMIT}"
            --output_path "${OUTPUT_DIR}/results.json"
            --log_samples
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
    TASK_TAG="${task}"
    case "${task}" in
        mmlu|cmmlu|ceval-valid|gpqa_main_n_shot)
            TASK_TAG="${task}_chat"
            ;;
    esac
    echo "Task: ${task}"
    for model_type in "${MODEL_TYPES[@]}"; do
        RESULT_FILE="results/${model_type}/${TASK_TAG}_${IMPORTANCE_TAG}/results.json"
        if [ -f "$RESULT_FILE" ]; then
            echo "  ✅ ${model_type}: results/${model_type}/${TASK_TAG}_${IMPORTANCE_TAG}/"
        else
            echo "  ❌ ${model_type}: FAILED"
        fi
    done
    echo ""
done

