#!/bin/bash
# Quick test script for GSM8K and HumanEval
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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

# Activate environment
source ~/miniconda3/bin/activate adaptive-dllm

cd "${PROJECT_ROOT}/evaluation/llada"

# Model configuration
MODEL_PATH=${MODEL_PATH:-"/data/qh_models/LLaDA-1.5"}
MODEL_NAME=${MODEL_NAME:-"llada_1_5"}
# Model types to run (can be overridden without editing file):
#   MODEL_TYPES_STR="standard,sparse,adaptive" bash run_eval_task.sh
MODEL_TYPES=("adaptive" "sparse")
if [ -n "${MODEL_TYPES_STR:-}" ]; then
    IFS=',' read -r -a MODEL_TYPES <<< "${MODEL_TYPES_STR}"
fi

# Output root (all results go here)
RESULTS_ROOT="${PROJECT_ROOT}/evaluation/llada/${MODEL_NAME}_results"
mkdir -p "$RESULTS_ROOT"

# -------------------------
# Importance score path selection (EDIT ONE LINE)
# -------------------------
# Only change the next line (or override via env IMPORTANCE_PATH=...):

IMPORTANCE_PATH=${IMPORTANCE_PATH:-"${PROJECT_ROOT}/configs/head_importance_llada-1_5_mmlu_all_pmrandom_threshold_ts20260116_025153/head_importance.pt"}
TASKS=("mmlu")
LIMIT=${LIMIT:-10}
# Whether to negate the scores (0=use original, 1=negate). Keep default aligned with prior behavior.
USE_NEGATED=${USE_NEGATED:-0}

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

# Generation parameters
GEN_LENGTH=256
STEPS=256
BLOCK_LENGTH=32
BLOCK_SIZE=32
DIFFUSION_MODE=${DIFFUSION_MODE:-"semi"}


# RULER parameters
# - RULER_LEN_K: max prompt length in K tokens (approx K*1024). Example: 4,8,16.
# - Provide either:
#   - RULER_DATA_PATH: JSONL file or directory (preferred; avoids HF network)
#   - or RULER_HF_DATASET (+ optional RULER_HF_CONFIG, RULER_SPLIT)
RULER_LEN_K=${RULER_LEN_K:-8}
RULER_LIMIT=${RULER_LIMIT:-$LIMIT}
RULER_DATA_PATH=${RULER_DATA_PATH:-""}
RULER_HF_DATASET=${RULER_HF_DATASET:-""}
RULER_HF_CONFIG=${RULER_HF_CONFIG:-""}
RULER_SPLIT=${RULER_SPLIT:-"validation"}
GPQA_DATA_PATH=${GPQA_DATA_PATH:-""}
LOCAL_TASK_ROOT="${PROJECT_ROOT}/evaluation/local_tasks/generated"
DEFAULT_GPQA_DATA_PATH="${PROJECT_ROOT}/evaluation/local_data/gpqa/gpqa_main.jsonl"

# Default to local exported JSONL if present
if [ -z "$RULER_DATA_PATH" ] && [ -d "/data/qh_models/ruler/jsonl/${RULER_SPLIT}" ]; then
    RULER_DATA_PATH="/data/qh_models/ruler/jsonl/${RULER_SPLIT}"
fi
if [ -z "${GPQA_DATA_PATH}" ] && [ -f "${DEFAULT_GPQA_DATA_PATH}" ]; then
    GPQA_DATA_PATH="${DEFAULT_GPQA_DATA_PATH}"
fi

# Tasks to run (can be overridden without editing file):
#   TASKS_STR="mmlu,ruler" bash run_eval_task.sh
if [ -n "${TASKS_STR:-}" ]; then
    IFS=',' read -r -a TASKS <<< "${TASKS_STR}"
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

echo "========================================================"
echo "Quick Test Configuration"
echo "========================================================"
echo "Tasks: ${TASKS[*]}"
echo "Model Types: ${MODEL_TYPES[*]}"
echo "Default Gen Length/Steps: ${GEN_LENGTH}/${STEPS} (task overrides apply for minerva_math and mbpp)"
echo "Block Length: ${BLOCK_LENGTH}, Block Size: ${BLOCK_SIZE}, Diffusion Mode: ${DIFFUSION_MODE}"
echo "Limit: ${LIMIT}"
echo "RULER len_k (if enabled): ${RULER_LEN_K}k"
echo "Importance tag: ${IMPORTANCE_TAG}"
echo "Importance base: ${IMPORTANCE_PATH}"
echo "Importance used: ${USED_IMPORTANCE_PATH}"
echo "========================================================"
echo ""

# Validate RULER inputs once (avoid repeating the same error for each model type)
NEEDS_RULER=0
for t in "${TASKS[@]}"; do
    if [ "$t" = "ruler" ]; then
        NEEDS_RULER=1
        break
    fi
done
if [ "$NEEDS_RULER" -eq 1 ]; then
    if [ -z "$RULER_DATA_PATH" ] && [ -z "$RULER_HF_DATASET" ]; then
        echo "ERROR: RULER requires RULER_DATA_PATH (JSONL) or RULER_HF_DATASET."
        echo "Example (local jsonl): TASKS_STR=ruler RULER_DATA_PATH=/path/to/ruler.jsonl RULER_LEN_K=8 bash run_eval_task.sh"
        echo "Example (local dir):   TASKS_STR=ruler RULER_DATA_PATH=/path/to/ruler_dir  RULER_LEN_K=8 bash run_eval_task.sh"
        echo "Example (HF dataset):  TASKS_STR=ruler RULER_HF_DATASET=ORG/NAME RULER_SPLIT=test RULER_LEN_K=8 bash run_eval_task.sh"
        exit 2
    fi
fi

# Function to run evaluation for one model type on one task
run_single_eval() {
    local task=$1
    local model_type=$2
    local task_name="${task}"
    
    echo ""
    echo "========================================"
    echo "Running: ${model_type} on ${task}"
    echo "========================================"
    
    local task_tag="${task}"
    if [ "$task" = "ruler" ]; then
        task_tag="ruler_${RULER_LEN_K}k"
    fi

    OUTPUT_DIR="${RESULTS_ROOT}/${model_type}/${task_tag}_${IMPORTANCE_TAG}"
    mkdir -p "$OUTPUT_DIR"
    local progress_label="${task}|${model_type}|${task_tag}_${IMPORTANCE_TAG}"
    
    # Task-specific generation lengths (override global GEN_LENGTH/STEPS for tasks that need more tokens)
    local local_gen_length=${GEN_LENGTH}
    local local_steps=${STEPS}
    case "$task" in
        mbpp)
            local_gen_length=${MBPP_GEN_LENGTH:-1024}
            local_steps=${MBPP_STEPS:-${local_gen_length}}
            ;;
        minerva_math)
            local_gen_length=${MINERVA_MATH_GEN_LENGTH:-1024}
            local_steps=${MINERVA_MATH_STEPS:-${local_gen_length}}
            ;;
    esac

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

    # Record start time
    START_TIME=$(date +%s)
    
    echo "Params: gen_length=${local_gen_length}, steps=${local_steps}, block_length=${BLOCK_LENGTH}, block_size=${BLOCK_SIZE}, diffusion_mode=${DIFFUSION_MODE}, limit=${LIMIT}"
    
    # Set importance source for adaptive mode
    if [ "$model_type" = "adaptive" ]; then
        IMPORTANCE_ARG=",importance_source=precomputed,precomputed_importance_path=${USED_IMPORTANCE_PATH}"
    else
        IMPORTANCE_ARG=""
    fi

    # Task-specific settings
    NUM_FEWSHOT=""
    MODEL_ARGS_EXTRA=""
    EVAL_BATCH_SIZE=""
    ENV_PREFIX=()
    INCLUDE_PATH_ARGS=()
    case "$task" in
        mmlu|cmmlu|ceval-valid|gpqa_main_n_shot)
            NUM_FEWSHOT=${MMLU_FEWSHOT:-5}
            if [ "$task" = "gpqa_main_n_shot" ]; then
                NUM_FEWSHOT=${GPQA_FEWSHOT:-5}
                if [ -n "${GPQA_DATA_PATH}" ]; then
                    GPQA_TASK_DIR="$(prepare_gpqa_local_task)" || return $?
                    INCLUDE_PATH_ARGS=(--include_path "${GPQA_TASK_DIR}")
                    task_name="gpqa_main_n_shot_local"
                    GPQA_LOCAL_ROWS="$(get_gpqa_local_row_count)" || return $?
                    GPQA_LOCAL_MAX_FEWSHOT=$(( GPQA_LOCAL_ROWS > 1 ? GPQA_LOCAL_ROWS - 1 : 0 ))
                    if [ "${NUM_FEWSHOT}" -gt "${GPQA_LOCAL_MAX_FEWSHOT}" ]; then
                        echo "[gpqa_local] Requested num_fewshot=${NUM_FEWSHOT}, but local dataset has only ${GPQA_LOCAL_ROWS} rows. Clamping to ${GPQA_LOCAL_MAX_FEWSHOT}."
                        NUM_FEWSHOT=${GPQA_LOCAL_MAX_FEWSHOT}
                    fi
                fi
            fi
            EVAL_BATCH_SIZE=1
            MODEL_ARGS_EXTRA=",mc_num=1,cfg=0.0,is_check_greedy=False,likelihood_now_step=${local_steps},recompute_mask_each_call=true"
            if [ "${MMLU_OFFLINE:-1}" = "1" ]; then
                ENV_PREFIX=(env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1)
            fi
            ;;
        ruler)
            ;;
        *)
            ;;
    esac
    
    # Build the command based on task
    if [ "$task" = "ruler" ]; then
        # RULER: run standalone evaluator (supports len_k).
        # Prefer local JSONL to avoid HF connectivity issues.
        RULER_ARGS=()
        if [ -n "$RULER_DATA_PATH" ]; then
            RULER_ARGS+=(--data_path "$RULER_DATA_PATH")
        elif [ -n "$RULER_HF_DATASET" ]; then
            RULER_ARGS+=(--hf_dataset "$RULER_HF_DATASET" --split "$RULER_SPLIT")
            if [ -n "$RULER_HF_CONFIG" ]; then
                RULER_ARGS+=(--hf_config "$RULER_HF_CONFIG")
            fi
        else
            echo "ERROR: RULER requires RULER_DATA_PATH (JSONL) or RULER_HF_DATASET."
            echo "Example: TASKS_STR=ruler RULER_DATA_PATH=/path/to/ruler.jsonl RULER_LEN_K=8 bash run_eval_task.sh"
            return 2
        fi

        python eval_ruler_llada.py \
            --model_path "${MODEL_PATH}" \
            --model_type "${model_type}" \
            --device cuda \
            --steps "${STEPS}" \
            --gen_length "${GEN_LENGTH}" \
            --block_length "${BLOCK_LENGTH}" \
            --diffusion_mode "${DIFFUSION_MODE}" \
            --skip 0.2 \
            --select 0.3 \
            --block_size "${BLOCK_SIZE}" \
            --importance_source precomputed \
            --precomputed_importance_path "${USED_IMPORTANCE_PATH}" \
            --len_k "${RULER_LEN_K}" \
            --limit "${RULER_LIMIT}" \
            --output_path "${OUTPUT_DIR}/results.json" \
            --samples_path "${OUTPUT_DIR}/samples.jsonl" \
            "${RULER_ARGS[@]}" \
            2>&1 | tee "${OUTPUT_DIR}/eval.log"
        CMD_RC=${PIPESTATUS[0]}
    elif [ "$task" = "humaneval" ] || [ "$task" = "mbpp" ]; then
        # Code-generation tasks require --confirm_run_unsafe_code.
        ${ENV_PREFIX[@]} python -m accelerate.commands.launch --num_processes=1 eval_llada.py \
            --model llada_eval \
            --model_args model_path="${MODEL_PATH}",model_type="${model_type}",gen_length=${local_gen_length},steps=${local_steps},block_length=${BLOCK_LENGTH},diffusion_mode=${DIFFUSION_MODE},skip=0.2,select=0.3,block_size=${BLOCK_SIZE},progress_label="${progress_label}"${IMPORTANCE_ARG}${MODEL_ARGS_EXTRA} \
            --tasks "${task_name}" \
            "${INCLUDE_PATH_ARGS[@]}" \
            ${NUM_FEWSHOT:+--num_fewshot ${NUM_FEWSHOT}} \
            ${EVAL_BATCH_SIZE:+--batch_size ${EVAL_BATCH_SIZE}} \
            --limit ${LIMIT} \
            --output_path "${OUTPUT_DIR}/results.json" \
            --log_samples \
            --confirm_run_unsafe_code \
            2>&1 | tee "${OUTPUT_DIR}/eval.log"
        CMD_RC=${PIPESTATUS[0]}
    else
        # GSM8K and other generation tasks
        ${ENV_PREFIX[@]} python -m accelerate.commands.launch --num_processes=1 eval_llada.py \
            --model llada_eval \
            --model_args model_path="${MODEL_PATH}",model_type="${model_type}",gen_length=${local_gen_length},steps=${local_steps},block_length=${BLOCK_LENGTH},diffusion_mode=${DIFFUSION_MODE},skip=0.2,select=0.3,block_size=${BLOCK_SIZE},progress_label="${progress_label}"${IMPORTANCE_ARG}${MODEL_ARGS_EXTRA} \
            --tasks "${task_name}" \
            "${INCLUDE_PATH_ARGS[@]}" \
            ${NUM_FEWSHOT:+--num_fewshot ${NUM_FEWSHOT}} \
            ${EVAL_BATCH_SIZE:+--batch_size ${EVAL_BATCH_SIZE}} \
            --limit ${LIMIT} \
            --output_path "${OUTPUT_DIR}/results.json" \
            --log_samples \
            2>&1 | tee "${OUTPUT_DIR}/eval.log"
        CMD_RC=${PIPESTATUS[0]}
    fi
    
    # Calculate running time
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    ELAPSED_MIN=$((ELAPSED / 60))
    ELAPSED_SEC=$((ELAPSED % 60))
    
    # Record time to file
    echo "${ELAPSED}" > "${OUTPUT_DIR}/runtime.txt"
    echo "$(date '+%Y-%m-%d %H:%M:%S') - ${model_type} - ${task} - ${ELAPSED}s (${ELAPSED_MIN}m ${ELAPSED_SEC}s)" >> "${RESULTS_ROOT}/timing_log.txt"
    
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
echo "🚀 Starting quick test evaluation..."
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
echo "📁 Results saved in: ${RESULTS_ROOT}/"
echo "📊 Timing log: ${RESULTS_ROOT}/timing_log.txt"
echo ""

# Generate a summary
echo "📈 Summary:"
echo ""
for task in "${TASKS[@]}"; do
    task_tag="${task}"
    if [ "$task" = "ruler" ]; then
        task_tag="ruler_${RULER_LEN_K}k"
    fi
    echo "Task: ${task}"
    for model_type in "${MODEL_TYPES[@]}"; do
        RESULT_FILE="${RESULTS_ROOT}/${model_type}/${task_tag}_${IMPORTANCE_TAG}/results.json"
        if [ -f "$RESULT_FILE" ]; then
            echo "  ✅ ${model_type}: ${RESULTS_ROOT}/${model_type}/${task_tag}_${IMPORTANCE_TAG}/"
        else
            echo "  ❌ ${model_type}: FAILED"
        fi
    done
    echo ""
done

