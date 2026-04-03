#!/bin/bash
# Dream head pruning / masking evaluation runner (lm-eval).
#
# 默认串行运行三种剪枝模式 (most / least / random)。
# 可通过 PRUNE_WHICH_LIST 指定子集，例如 PRUNE_WHICH_LIST="most,least"
#
# 剪枝粒度默认为 kv_group（与 adaptive sparse 的 gqa_weight_mode="kv" 对齐）。
# PRUNE_K / PRUNE_K_FRAC 指定剪枝的 unit 数/比例（kv_group 模式下 unit = group）。

set -o pipefail

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=/home/qiheng/Projects/adaptive-dllm:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4}

source ~/miniconda3/bin/activate adaptive-dllm

cd /home/qiheng/Projects/adaptive-dllm/evaluation/dream

# ===========================================================================
# Config
# ===========================================================================

# --- Model ---
MODEL_PATH=${MODEL_PATH:-"/data/qh_models/Dream-v0-Instruct-7B"}

# --- Pruning ---
PRUNE_WHICH_LIST=${PRUNE_WHICH_LIST:-"most,least,random"}
MASK_GRANULARITY=${MASK_GRANULARITY:-"kv_group"}  # kv_group | head
PRUNE_K=${PRUNE_K:-""}
PRUNE_K_FRAC=${PRUNE_K_FRAC:-"0.05"}
LAYER_START=${LAYER_START:-0}
LAYER_END=${LAYER_END:-27}
RANDOM_PRUNE_SEED=${RANDOM_PRUNE_SEED:-1234}
HEAD_MASK_WARMUP_FRAC=${HEAD_MASK_WARMUP_FRAC:-0.2}

if [ -z "$PRUNE_K" ] && [ -z "$PRUNE_K_FRAC" ]; then
    PRUNE_K_FRAC="0.25"
    echo "[config] PRUNE_K / PRUNE_K_FRAC not set. Defaulting to PRUNE_K_FRAC=${PRUNE_K_FRAC}."
fi
if [ -n "$PRUNE_K" ] && [ -n "$PRUNE_K_FRAC" ]; then
    echo "ERROR: set only one of PRUNE_K or PRUNE_K_FRAC (not both)."
    exit 2
fi

# --- Importance ---
IMPORTANCE_PATH=${IMPORTANCE_PATH:-"/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream_mmlu_all_pmrandom_threshold_ts20260323_224941/head_importance.pt"}
USE_NEGATED=${USE_NEGATED:-0}

# --- Tasks ---
TASKS=("mmlu")
if [ -n "${TASKS_STR:-}" ]; then
    IFS=',' read -r -a TASKS <<< "${TASKS_STR}"
fi
LIMIT=${LIMIT:-20}
GPQA_DATA_PATH=${GPQA_DATA_PATH:-""}
LOCAL_TASK_ROOT="/home/qiheng/Projects/adaptive-dllm/evaluation/local_tasks/generated"
DEFAULT_GPQA_DATA_PATH="/home/qiheng/Projects/adaptive-dllm/evaluation/local_data/gpqa/gpqa_main.jsonl"
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
    python "/home/qiheng/Projects/adaptive-dllm/evaluation/local_tasks/prepare_gpqa_local_task.py" \
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

USED_IMPORTANCE_PATH="${IMPORTANCE_PATH}"
if [ "${USE_NEGATED}" = "1" ]; then
    SRC_IMPORTANCE_PATH="${IMPORTANCE_PATH}"
    NEG_DIR=${NEG_DIR:-"$(dirname "${SRC_IMPORTANCE_PATH}")_neg"}
    USED_IMPORTANCE_PATH="${NEG_DIR}/head_importance.pt"
    if [ ! -f "$USED_IMPORTANCE_PATH" ]; then
        echo "[prep] Generating negated importance..."
        python /home/qiheng/Projects/adaptive-dllm/evaluation/dream/generate_negated_importance.py \
            --in_pt "$SRC_IMPORTANCE_PATH" \
            --out_dir "$NEG_DIR"
    fi
fi

DEFAULT_IMPORTANCE_TAG="$(basename "$(dirname "${IMPORTANCE_PATH}")")$( [ "${USE_NEGATED}" = "1" ] && echo "_neg" )"
IMPORTANCE_TAG=${IMPORTANCE_TAG:-"${DEFAULT_IMPORTANCE_TAG}"}


# --- Generation params (match official Dream eval) ---
TEMPERATURE=${TEMPERATURE:-0.1}
TOP_P=${TOP_P:-0.9}
ALG=${ALG:-"entropy"}
ALG_TEMP=${ALG_TEMP:-0.0}
BLOCK_SIZE=${BLOCK_SIZE:-32}
BLOCK_LENGTH=${BLOCK_LENGTH:-32}
DIFFUSION_MODE=${DIFFUSION_MODE:-"global"}

# --- Output ---
RESULTS_ROOT="/home/qiheng/Projects/adaptive-dllm/evaluation/dream/results/mask_head"
mkdir -p "$RESULTS_ROOT"

RUN_TS=$(date +"%Y-%m-%dT%H-%M-%S")

# ===========================================================================
# Helpers
# ===========================================================================

run_single_eval() {
    local task=$1
    local prune_which=$2
    local run_dir=$3
    local out_dir="${run_dir}/${task}"
    local task_name="${task}"
    local progress_label="${task}|mask_head|$(basename "${run_dir}")"
    mkdir -p "$out_dir"

    local max_new_tokens=256
    local steps=256
    local apply_chat=""
    local num_fewshot=""
    local eval_batch_size=""
    local model_args_extra=""
    local env_prefix=()

    if [ "$task" = "humaneval" ] || [ "$task" = "humaneval_instruct" ]; then
        max_new_tokens=768
        steps=768
    elif [ "$task" = "mbpp" ]; then
        max_new_tokens=${MBPP_MAX_NEW_TOKENS:-512}
        steps=${MBPP_STEPS:-${max_new_tokens}}
    elif [ "$task" = "minerva_math" ]; then
        max_new_tokens=${MINERVA_MATH_MAX_NEW_TOKENS:-512}
        steps=${MINERVA_MATH_STEPS:-${max_new_tokens}}
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

    if [ "$task" = "gsm8k" ] || [ "$task" = "gsm8k_cot" ] || [ "$task" = "minerva_math" ]; then
        apply_chat="--apply_chat_template"
        num_fewshot=${GSM8K_FEWSHOT:-0}
    fi
    if [ "$task" = "mmlu" ] || [ "$task" = "cmmlu" ] || [ "$task" = "ceval-valid" ] || [ "$task" = "gpqa_main_n_shot" ]; then
        apply_chat="--apply_chat_template"
        if [ "$task" = "gpqa_main_n_shot" ]; then
            num_fewshot=${GPQA_FEWSHOT:-5}
            if [ -n "${GPQA_DATA_PATH}" ]; then
                GPQA_LOCAL_ROWS="$(get_gpqa_local_row_count)" || return $?
                GPQA_LOCAL_MAX_FEWSHOT=$(( GPQA_LOCAL_ROWS > 1 ? GPQA_LOCAL_ROWS - 1 : 0 ))
                if [ "${num_fewshot}" -gt "${GPQA_LOCAL_MAX_FEWSHOT}" ]; then
                    echo "[gpqa_local] Requested num_fewshot=${num_fewshot}, but local dataset has only ${GPQA_LOCAL_ROWS} rows. Clamping to ${GPQA_LOCAL_MAX_FEWSHOT}."
                    num_fewshot=${GPQA_LOCAL_MAX_FEWSHOT}
                fi
            fi
        else
            num_fewshot=${MMLU_FEWSHOT:-5}
        fi
        eval_batch_size=1
        model_args_extra=",mc_num=1,likelihood_now_step=${steps},recompute_mask_each_call=true"
        if [ "${MMLU_OFFLINE:-1}" = "1" ]; then
            env_prefix=(env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1)
        fi
    fi
    local include_path_args=()
    if [ "$task" = "gpqa_main_n_shot" ] && [ -n "${GPQA_DATA_PATH}" ]; then
        GPQA_TASK_DIR="$(prepare_gpqa_local_task)" || return $?
        include_path_args=(--include_path "${GPQA_TASK_DIR}")
        task_name="gpqa_main_n_shot_local"
    fi

    local prune_args=""
    if [ -n "$PRUNE_K" ]; then
        prune_args=",prune_k=${PRUNE_K}"
    else
        prune_args=",prune_k_frac=${PRUNE_K_FRAC}"
    fi

    local importance_arg=""
    if [ "$prune_which" != "random" ]; then
        importance_arg=",importance_path=${USED_IMPORTANCE_PATH}"
    fi

    local model_args="model_path=${MODEL_PATH},max_new_tokens=${max_new_tokens},steps=${steps},temperature=${TEMPERATURE},top_p=${TOP_P},alg=${ALG},alg_temp=${ALG_TEMP},diffusion_mode=${DIFFUSION_MODE},block_length=${BLOCK_LENGTH},block_size=${BLOCK_SIZE},progress_label=${progress_label}${model_args_extra}${importance_arg},prune_which=${prune_which}${prune_args},random_prune_seed=${RANDOM_PRUNE_SEED},layer_start=${LAYER_START},layer_end=${LAYER_END},mask_granularity=${MASK_GRANULARITY},head_mask_warmup_frac=${HEAD_MASK_WARMUP_FRAC}"

    if [ "$task" = "humaneval" ] || [ "$task" = "mbpp" ]; then
        ${env_prefix[@]} python -m accelerate.commands.launch --num_processes=1 eval_mask_head_dream.py \
            --model dream_mask_head_eval \
            --model_args "${model_args}" \
            --tasks "${task_name}" \
            "${include_path_args[@]}" \
            --num_fewshot 0 \
            --limit ${LIMIT} \
            --output_path "${out_dir}/results.json" \
            --log_samples \
            --confirm_run_unsafe_code \
            2>&1 | tee "${out_dir}/eval.log"
        return ${PIPESTATUS[0]}
    else
        ${env_prefix[@]} python -m accelerate.commands.launch --num_processes=1 eval_mask_head_dream.py \
            --model dream_mask_head_eval \
            --model_args "${model_args}" \
            --tasks "${task_name}" \
            "${include_path_args[@]}" \
            ${num_fewshot:+--num_fewshot ${num_fewshot}} \
            ${eval_batch_size:+--batch_size ${eval_batch_size}} \
            ${apply_chat} \
            --limit ${LIMIT} \
            --output_path "${out_dir}/results.json" \
            --log_samples \
            2>&1 | tee "${out_dir}/eval.log"
        return ${PIPESTATUS[0]}
    fi
}

# ===========================================================================
# Main: loop over prune modes
# ===========================================================================

IFS=',' read -r -a PRUNE_MODES <<< "${PRUNE_WHICH_LIST}"

echo "========================================================"
echo "Dream Mask-Head / Pruning Eval"
echo "========================================================"
echo "Model:       ${MODEL_PATH}"
echo "Granularity: ${MASK_GRANULARITY}"
echo "Prune modes: ${PRUNE_MODES[*]}"
echo "Prune frac:  k=${PRUNE_K:-"(none)"} k_frac=${PRUNE_K_FRAC:-"(none)"}  layers=${LAYER_START}..${LAYER_END}"
echo "Diffusion:   mode=${DIFFUSION_MODE} block_length=${BLOCK_LENGTH}"
echo "HeadMask:    warmup_frac=${HEAD_MASK_WARMUP_FRAC}"
echo "Seed:        ${RANDOM_PRUNE_SEED} (random mode only)"
echo "Importance:  ${USED_IMPORTANCE_PATH} (tag=${IMPORTANCE_TAG})"
echo "Tasks:       ${TASKS[*]}"
echo "Limit:       ${LIMIT}"
echo "Timestamp:   ${RUN_TS}"
echo "========================================================"

FAIL=0

for prune_which in "${PRUNE_MODES[@]}"; do
    # Validate importance file for non-random modes
    if [ "$prune_which" != "random" ]; then
        if [ ! -f "$USED_IMPORTANCE_PATH" ]; then
            echo "ERROR: importance file not found: ${USED_IMPORTANCE_PATH}"
            exit 3
        fi
    fi

    # Build per-mode output directory
    if [ -n "$PRUNE_K" ]; then
        PRUNE_TAG="prune_${prune_which}_k${PRUNE_K}"
    else
        PRUNE_TAG="prune_${prune_which}_kfrac$(echo "${PRUNE_K_FRAC}" | tr '.' 'p')"
    fi
    MODE_DIR="${RESULTS_ROOT}/${IMPORTANCE_TAG}/${PRUNE_TAG}_${MASK_GRANULARITY}_L${LAYER_START}-${LAYER_END}_${RUN_TS}"
    mkdir -p "$MODE_DIR"

    echo ""
    echo "========================================"
    echo "Prune mode: ${prune_which}  ->  ${MODE_DIR}"
    echo "========================================"

    for task in "${TASKS[@]}"; do
        echo ""
        echo "--- ${prune_which} / ${task} ---"
        if run_single_eval "$task" "$prune_which" "$MODE_DIR"; then
            echo "[ok] ${prune_which} / ${task}"
        else
            echo "[FAIL] ${prune_which} / ${task}  (see ${MODE_DIR}/${task}/eval.log)"
            FAIL=1
        fi
    done
done

echo ""
echo "========================================================"
if [ "$FAIL" -eq 0 ]; then
    echo "All evaluations completed successfully."
else
    echo "Some evaluations failed. Check logs above."
fi
echo "========================================================"

exit $FAIL
