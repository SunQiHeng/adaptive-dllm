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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Activate the same environment used by the attribution and pruning runners.
# This is required for non-interactive/background jobs, whose PATH does not
# otherwise contain python/accelerate.
source ~/miniconda3/bin/activate adaptive-dllm

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

# Negation selection:
# - Backward compatible: USE_NEGATED=0 or USE_NEGATED=1
# - Batch mode: USE_NEGATED_MODES_STR="0,1" runs both original and negated scores in one invocation
USE_NEGATED=${USE_NEGATED:-1}
USE_NEGATED_MODES_STR=${USE_NEGATED_MODES_STR:-"${USE_NEGATED}"}
IFS=',' read -r -a USE_NEGATED_MODES <<< "${USE_NEGATED_MODES_STR}"
IMPORTANCE_TAG_OVERRIDE=${IMPORTANCE_TAG:-""}

# Resolved per negation mode before each run batch.
USED_IMPORTANCE_PATH="${IMPORTANCE_PATH}"
IMPORTANCE_TAG=""

# -------------------------
# Importance score path selection
# -------------------------
# Preferred interface:
#   Single dataset: MODEL_NAME=dream ATTR_METHOD=headig ATTR_DATASETS_STR=ceval-valid_all
#   Multi dataset:  MODEL_NAME=dream ATTR_METHOD=headig ATTR_DATASETS_STR="mmlu_all,cmmlu_all"
# This resolves to:
#   ${PROJECT_ROOT}/configs/dream_headig_ceval-valid_all/head_importance.pt
#
# You can still override everything directly with IMPORTANCE_PATH=...
USER_IMPORTANCE_PATH=${IMPORTANCE_PATH:-""}
ATTR_MODEL_NAME=${MODEL_NAME:-"dream"}
# ATTR_METHOD candidates:
#   headig | attnlrp | shapley | attarr | loo
ATTR_METHOD=${ATTR_METHOD:-"shapley"}
# ATTR_DATASETS_STR candidates:
#   mmlu_all | cmmlu_all | ceval-valid_all | gsm8k | minerva_math | gpqa_main_n_shot_all | humaneval | mbpp
ATTR_DATASETS_STR=${ATTR_DATASETS_STR:-"mmlu_all,cmmlu_all,ceval-valid_all,gsm8k,minerva_math,gpqa_main_n_shot_all,humaneval,mbpp"}
IFS=',' read -r -a ATTR_DATASETS <<< "${ATTR_DATASETS_STR}"
FIRST_ATTR_DATASET="$(echo "${ATTR_DATASETS[0]}" | xargs)"

build_default_importance_path() {
    local attr_dataset="$1"
    echo "${PROJECT_ROOT}/configs/${ATTR_MODEL_NAME}_${ATTR_METHOD}_${attr_dataset}/head_importance.pt"
}

build_aconfig_importance_path() {
    local attr_dataset="$1"
    python - "$PROJECT_ROOT" "$ATTR_MODEL_NAME" "$ATTR_METHOD" "$attr_dataset" <<'PY'
from pathlib import Path
import sys

project_root = Path(sys.argv[1])
model_name = sys.argv[2]
attr_method = sys.argv[3]
attr_dataset = sys.argv[4]
root = project_root / "configs" / "aconfigs"
if not root.is_dir():
    print("")
    raise SystemExit(0)

if attr_method == "headig":
    def matches(name: str) -> bool:
        return name.startswith(f"head_importance_{model_name}_{attr_dataset}_pm")
elif attr_method == "attnlrp":
    def matches(name: str) -> bool:
        return name.startswith(f"head_importance_{model_name}_{attr_dataset}_attnlrp_")
elif attr_method == "shapley":
    def matches(name: str) -> bool:
        return name.startswith(f"head_importance_{model_name}_{attr_dataset}_shapley_")
elif attr_method == "attarr":
    def matches(name: str) -> bool:
        return name.startswith(f"head_importance_{model_name}_{attr_dataset}_attarr_")
elif attr_method == "loo":
    def matches(name: str) -> bool:
        return name.startswith(f"head_importance_{model_name}_{attr_dataset}_loo_")
else:
    print("")
    raise SystemExit(0)

candidates = [p for p in root.iterdir() if p.is_dir() and matches(p.name) and (p / "head_importance.pt").is_file()]
candidates.sort(key=lambda p: (p.stat().st_mtime, p.name), reverse=True)
print(str(candidates[0] / "head_importance.pt") if candidates else "")
PY
}

resolve_auto_importance_path() {
    local attr_dataset="$1"
    local legacy_path
    legacy_path="$(build_default_importance_path "${attr_dataset}")"
    if [ -f "${legacy_path}" ]; then
        echo "${legacy_path}"
        return 0
    fi

    local aconfig_path
    aconfig_path="$(build_aconfig_importance_path "${attr_dataset}")"
    if [ -n "${aconfig_path}" ] && [ -f "${aconfig_path}" ]; then
        echo "${aconfig_path}"
        return 0
    fi

    echo "${legacy_path}"
}

default_task_for_attr_dataset() {
    case "$1" in
        mmlu_all) echo "mmlu" ;;
        cmmlu_all) echo "cmmlu" ;;
        ceval-valid_all) echo "ceval-valid" ;;
        gsm8k) echo "gsm8k" ;;
        minerva_math) echo "minerva_math" ;;
        gpqa_main_n_shot_all) echo "gpqa_main_n_shot" ;;
        humaneval) echo "humaneval" ;;
        mbpp) echo "mbpp" ;;
        *)
            echo "ERROR: Unsupported attr dataset for default task mapping: $1" >&2
            return 2
            ;;
    esac
}

# Recommended LIMIT values for task-specific evals (stability vs runtime trade-off):
#   mmlu_all: 40
#   cmmlu_all: 40
#   ceval-valid_all: 200
#   gpqa_main_n_shot_all: 200
#   gsm8k: 200
#   minerva_math: 200
#   humaneval: 200
#   mbpp: 200
default_limit_for_attr_dataset() {
    case "$1" in
        mmlu_all) echo 40 ;;
        cmmlu_all) echo 40 ;;
        ceval-valid_all) echo 200 ;;
        gpqa_main_n_shot_all) echo 200 ;;
        gsm8k) echo 200 ;;
        minerva_math) echo "${MINERVA_LIMIT_PER_SUBTASK:-29}" ;;
        humaneval) echo 200 ;;
        mbpp) echo 200 ;;
        *)
            echo "ERROR: Unsupported attr dataset for default limit mapping: $1" >&2
            return 2
            ;;
    esac
}

resolve_attr_dataset_context() {
    local attr_dataset="$(echo "$1" | xargs)"
    CURRENT_ATTR_DATASET="${attr_dataset}"
    IMPORTANCE_PATH="${USER_IMPORTANCE_PATH:-$(resolve_auto_importance_path "${CURRENT_ATTR_DATASET}")}"
    if [ -n "${TASKS_STR:-}" ]; then
        IFS=',' read -r -a TASKS <<< "${TASKS_STR}"
    else
        TASKS=("$(default_task_for_attr_dataset "${CURRENT_ATTR_DATASET}")") || return $?
    fi
    LIMIT="${LIMIT_OVERRIDE:-$(default_limit_for_attr_dataset "${CURRENT_ATTR_DATASET}")}" || return $?
}

LIMIT_OVERRIDE=${LIMIT:-""}
CURRENT_ATTR_DATASET=""
resolve_attr_dataset_context "${FIRST_ATTR_DATASET}" || exit $?
GPQA_DATA_PATH=${GPQA_DATA_PATH:-""}
LOCAL_TASK_ROOT="${PROJECT_ROOT}/evaluation/local_tasks/generated"
DEFAULT_GPQA_DATA_PATH="${PROJECT_ROOT}/evaluation/local_data/gpqa/gpqa_main.jsonl"
if [ -z "${GPQA_DATA_PATH}" ] && [ -f "${DEFAULT_GPQA_DATA_PATH}" ]; then
    GPQA_DATA_PATH="${DEFAULT_GPQA_DATA_PATH}"
fi
export GPQA_DATA_PATH

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


resolve_importance_variant() {
    local use_negated="$1"
    local tag_suffix=""

    USED_IMPORTANCE_PATH="${IMPORTANCE_PATH}"
    if [ "${use_negated}" = "1" ]; then
        local src_importance_path="${IMPORTANCE_PATH}"
        local neg_dir="${NEG_DIR:-"$(dirname "${src_importance_path}")_neg"}"
        USED_IMPORTANCE_PATH="${neg_dir}/head_importance.pt"
        if [ ! -f "${USED_IMPORTANCE_PATH}" ]; then
            echo "➖ Generating negated importance..."
            python "${SCRIPT_DIR}/generate_negated_importance.py" \
                --in_pt "${src_importance_path}" \
                --out_dir "${neg_dir}"
            if [ ! -f "${USED_IMPORTANCE_PATH}" ]; then
                echo "ERROR: Failed to generate negated importance at: ${USED_IMPORTANCE_PATH}"
                exit 3
            fi
        else
            echo "➖ Using existing negated importance: ${USED_IMPORTANCE_PATH}"
        fi
        tag_suffix="_neg"
    fi

    local default_importance_tag
    default_importance_tag="$(basename "$(dirname "${IMPORTANCE_PATH}")")${tag_suffix}"
    if [ -n "${IMPORTANCE_TAG_OVERRIDE}" ]; then
        if [ "${#USE_NEGATED_MODES[@]}" -gt 1 ]; then
            if [ "${use_negated}" = "1" ]; then
                IMPORTANCE_TAG="${IMPORTANCE_TAG_OVERRIDE}_neg"
            else
                IMPORTANCE_TAG="${IMPORTANCE_TAG_OVERRIDE}_orig"
            fi
        else
            IMPORTANCE_TAG="${IMPORTANCE_TAG_OVERRIDE}"
        fi
    else
        IMPORTANCE_TAG="${default_importance_tag}"
    fi
}

echo "========================================================"
echo "Dream Quick Test Configuration"
echo "========================================================"
echo "Attr datasets: ${ATTR_DATASETS[*]}"
echo "Model Types: ${MODEL_TYPES[*]}"
echo "Negation modes: ${USE_NEGATED_MODES[*]}"
echo "Default Max New Tokens/Steps: 256/256 (task overrides apply for humaneval, mbpp, and minerva_math)"
echo "Block Size: 32"
if [ -n "${USER_IMPORTANCE_PATH}" ]; then
    echo "Importance base: ${USER_IMPORTANCE_PATH} (manual override for all datasets)"
else
    echo "Importance base: auto-resolved per item in ATTR_DATASETS_STR"
fi
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
DIFFUSION_MODE=${DIFFUSION_MODE:-"semi"}


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
        MAX_NEW_TOKENS=${HUMANEVAL_MAX_NEW_TOKENS:-768}
        STEPS=${HUMANEVAL_STEPS:-${MAX_NEW_TOKENS}}
    elif [ "$task" = "mbpp" ]; then
        MAX_NEW_TOKENS=${MBPP_MAX_NEW_TOKENS:-1024}
        STEPS=${MBPP_STEPS:-${MAX_NEW_TOKENS}}
    elif [ "$task" = "gsm8k" ] || [ "$task" = "gsm8k_cot" ]; then
        MAX_NEW_TOKENS=256
        STEPS=256
    elif [ "$task" = "minerva_math" ]; then
        MAX_NEW_TOKENS=${MINERVA_MATH_MAX_NEW_TOKENS:-1024}
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
    
    echo "[start] ${CURRENT_RUN_LABEL} max_new_tokens=${MAX_NEW_TOKENS} steps=${STEPS} limit=${LIMIT} out=${OUTPUT_DIR}"
    
    # Fail early with a clear error if adaptive mode resolves to a missing importance file.
    if [ "$model_type" = "adaptive" ] && [ ! -f "${USED_IMPORTANCE_PATH}" ]; then
        echo "ERROR: adaptive mode requires an importance file, but it was not found:"
        echo "  ${USED_IMPORTANCE_PATH}"
        return 3
    fi

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
            --model_args "${MODEL_ARGS_STR},mc_num=${MC_NUM:-1},likelihood_now_step=${STEPS},recompute_mask_each_call=true"
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
        echo "[done]  ${CURRENT_RUN_LABEL} elapsed=${ELAPSED}s"
    else
        echo "[fail]  ${CURRENT_RUN_LABEL} exit=${CMD_RC} elapsed=${ELAPSED}s log=${OUTPUT_DIR}/eval.log"
    fi

    return "${CMD_RC}"
}

# Main execution
echo "🚀 Starting Dream quick test evaluation..."
echo "Started at: $(date)"
echo ""

# Total tasks counter
if [ -n "${TASKS_STR:-}" ]; then
    IFS=',' read -r -a TASKS_OVERRIDE <<< "${TASKS_STR}"
    TASKS_PER_DATASET=${#TASKS_OVERRIDE[@]}
else
    TASKS_PER_DATASET=1
fi
TOTAL_TASKS=$((${#ATTR_DATASETS[@]} * ${TASKS_PER_DATASET} * ${#MODEL_TYPES[@]} * ${#USE_NEGATED_MODES[@]}))
CURRENT_TASK=0

# Run all combinations
for attr_dataset in "${ATTR_DATASETS[@]}"; do
    resolve_attr_dataset_context "${attr_dataset}" || exit $?
    echo "[dataset] attr_dataset=${CURRENT_ATTR_DATASET} tasks=${TASKS[*]} limit=${LIMIT}"
    for use_negated_mode in "${USE_NEGATED_MODES[@]}"; do
        use_negated_mode="$(echo "${use_negated_mode}" | xargs)"
        if [ "${use_negated_mode}" != "0" ] && [ "${use_negated_mode}" != "1" ]; then
            echo "ERROR: USE_NEGATED_MODES_STR only supports 0 or 1, got: ${use_negated_mode}"
            exit 2
        fi

        resolve_importance_variant "${use_negated_mode}"
        negation_label="$( [ "${use_negated_mode}" = "1" ] && echo "negated" || echo "original" )"
        echo "[importance] variant=${negation_label} tag=${IMPORTANCE_TAG} path=${USED_IMPORTANCE_PATH}"
        for task in "${TASKS[@]}"; do
            for model_type in "${MODEL_TYPES[@]}"; do
                CURRENT_TASK=$((CURRENT_TASK + 1))
                CURRENT_RUN_LABEL="[${CURRENT_TASK}/${TOTAL_TASKS}] dataset=${CURRENT_ATTR_DATASET} variant=${negation_label} task=${task} model=${model_type}"
                run_single_eval "$task" "$model_type" || exit $?
            done
        done
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
for attr_dataset in "${ATTR_DATASETS[@]}"; do
    resolve_attr_dataset_context "${attr_dataset}" || exit $?
    echo "Attr dataset: ${CURRENT_ATTR_DATASET} (tasks=${TASKS[*]}, limit=${LIMIT})"
    for use_negated_mode in "${USE_NEGATED_MODES[@]}"; do
        use_negated_mode="$(echo "${use_negated_mode}" | xargs)"
        resolve_importance_variant "${use_negated_mode}"
        echo "Importance variant: $( [ "${use_negated_mode}" = "1" ] && echo "negated" || echo "original" )"
        for task in "${TASKS[@]}"; do
            TASK_TAG="${task}"
            case "${task}" in
                mmlu|cmmlu|ceval-valid|gpqa_main_n_shot)
                    TASK_TAG="${task}_chat"
                    ;;
            esac
            echo "Task: ${task}"
            for model_type in "${MODEL_TYPES[@]}"; do
                RESULT_DIR="results/${model_type}/${TASK_TAG}_${IMPORTANCE_TAG}"
                if compgen -G "${RESULT_DIR}/results*.json" > /dev/null; then
                    echo "  ✅ ${model_type}: ${RESULT_DIR}/"
                else
                    echo "  ❌ ${model_type}: FAILED"
                fi
            done
            echo ""
        done
    done
done
