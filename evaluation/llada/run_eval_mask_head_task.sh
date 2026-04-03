#!/bin/bash
# Head pruning / masking evaluation runner for LLaDA (lm-eval).
#
# 目标：
# - 根据 head importance 分数剪枝：剪最重要 top-k 或最不重要 top-k
# - 跑 lm-eval 任务（gsm8k / humaneval / mmlu 等）
#
# 用法示例：
#   MODES="most,least,random" bash run_eval_mask_head_task.sh

set -o pipefail

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=/home/qiheng/Projects/adaptive-dllm:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

source ~/miniconda3/bin/activate adaptive-dllm

cd /home/qiheng/Projects/adaptive-dllm/evaluation/llada

# -----------------------
# Model config
# -----------------------
MODEL_PATH=${MODEL_PATH:-"/data/qh_models/LLaDA-1.5"}
MODEL_NAME=${MODEL_NAME:-"llada_1_5"}

# -----------------------
# Pruning / Importance config
# -----------------------
# 现在支持一次运行多个模式，用逗号分隔，例如: MODES="most,least,random"
MODES_STR=${MODES:-"most"}
IFS=',' read -r -a MODES <<< "${MODES_STR}"

PRUNE_SCOPE=${PRUNE_SCOPE:-"layer"} # global|layer (全局排序剪枝 vs 每层按相同比例剪枝)
# 修改这里即可更换分数路径
IMPORTANCE_PATH=${IMPORTANCE_PATH:-"/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada-1_5_gsm8k_full_ts20260115_024826/head_importance.pt"}

PRUNE_K_FRAC=${PRUNE_K_FRAC:-"0.2"}
LAYER_START=${LAYER_START:-0}
LAYER_END=${LAYER_END:-31}
RANDOM_PRUNE_SEED=${RANDOM_PRUNE_SEED:-1234}
HEAD_MASK_WARMUP_FRAC=${HEAD_MASK_WARMUP_FRAC:-0.2}

# -----------------------
# Eval tasks / params
# -----------------------
TASKS=("gsm8k")
if [ -n "${TASKS_STR:-}" ]; then
    IFS=',' read -r -a TASKS <<< "${TASKS_STR}"
fi
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

GEN_LENGTH=${GEN_LENGTH:-256}
STEPS=${STEPS:-256}
BLOCK_LENGTH=${BLOCK_LENGTH:-32}
DIFFUSION_MODE=${DIFFUSION_MODE:-"semi"}
LIMIT=${LIMIT:-20}
NUM_FEWSHOT=${NUM_FEWSHOT:-0}

# -----------------------
# Run Loop
# -----------------------
for current_mode in "${MODES[@]}"; do
    PRUNE_WHICH=$current_mode
    
    # -----------------------
    # Output config
    # -----------------------
    RESULTS_ROOT="/home/qiheng/Projects/adaptive-dllm/evaluation/llada/${MODEL_NAME}_results/mask_head"
    PRUNE_TAG="prune_${PRUNE_WHICH}_scope_${PRUNE_SCOPE}_kfrac$(echo "${PRUNE_K_FRAC}" | tr '.' 'p')"
    RUN_TS=$(date +"%Y-%m-%dT%H-%M-%S")
    RUN_DIR="${RESULTS_ROOT}/${PRUNE_TAG}_L${LAYER_START}-${LAYER_END}_${RUN_TS}"
    mkdir -p "$RUN_DIR"

    echo "========================================================"
    echo "LLaDA Mask-Head / Pruning Eval | MODE: ${PRUNE_WHICH}"
    echo "========================================================"
    echo "Model:       ${MODEL_PATH}"
    echo "Importance:  ${IMPORTANCE_PATH}"
    echo "Prune:       which=${PRUNE_WHICH} scope=${PRUNE_SCOPE} k_frac=${PRUNE_K_FRAC}"
    echo "Tasks:       ${TASKS[*]}"
    echo "Diffusion:   mode=${DIFFUSION_MODE} block_length=${BLOCK_LENGTH}"
    echo "HeadMask:    warmup_frac=${HEAD_MASK_WARMUP_FRAC}"
    echo "Out:         ${RUN_DIR}"
    echo "========================================================"

    run_single_eval() {
        local task=$1
        local out_dir="${RUN_DIR}/${task}"
        local task_name="${task}"
        local progress_label="${task}|mask_head|$(basename "${RUN_DIR}")"
        mkdir -p "$out_dir"

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

        local num_fewshot=""
        local eval_batch_size=""
        local model_args_extra=""
        local env_prefix=()
        local include_path_args=()

        case "$task" in
            mmlu|cmmlu|ceval-valid|gpqa_main_n_shot)
                num_fewshot=${MMLU_FEWSHOT:-5}
                if [ "$task" = "gpqa_main_n_shot" ]; then
                    num_fewshot=${GPQA_FEWSHOT:-5}
                    if [ -n "${GPQA_DATA_PATH}" ]; then
                        GPQA_TASK_DIR="$(prepare_gpqa_local_task)" || return $?
                        include_path_args=(--include_path "${GPQA_TASK_DIR}")
                        task_name="gpqa_main_n_shot_local"
                        GPQA_LOCAL_ROWS="$(get_gpqa_local_row_count)" || return $?
                        GPQA_LOCAL_MAX_FEWSHOT=$(( GPQA_LOCAL_ROWS > 1 ? GPQA_LOCAL_ROWS - 1 : 0 ))
                        if [ "${num_fewshot}" -gt "${GPQA_LOCAL_MAX_FEWSHOT}" ]; then
                            echo "[gpqa_local] Requested num_fewshot=${num_fewshot}, but local dataset has only ${GPQA_LOCAL_ROWS} rows. Clamping to ${GPQA_LOCAL_MAX_FEWSHOT}."
                            num_fewshot=${GPQA_LOCAL_MAX_FEWSHOT}
                        fi
                    fi
                fi
                eval_batch_size=1
                model_args_extra=",mc_num=1,cfg=0.0,is_check_greedy=False,likelihood_now_step=${local_steps},recompute_mask_each_call=true"
                if [ "${MMLU_OFFLINE:-1}" = "1" ]; then
                    env_prefix=(env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1)
                fi
                ;;
        esac

        local importance_arg=""
        if [ "${PRUNE_WHICH}" != "random" ]; then
            if [ ! -f "$IMPORTANCE_PATH" ]; then
                echo "ERROR: importance file not found: ${IMPORTANCE_PATH}"
                exit 3
            fi
            importance_arg=",importance_path=${IMPORTANCE_PATH}"
        fi

        local common_args="model_path=${MODEL_PATH}${importance_arg},prune_which=${PRUNE_WHICH},prune_k_frac=${PRUNE_K_FRAC},prune_scope=${PRUNE_SCOPE},random_prune_seed=${RANDOM_PRUNE_SEED},layer_start=${LAYER_START},layer_end=${LAYER_END},gen_length=${local_gen_length},steps=${local_steps},block_length=${BLOCK_LENGTH},diffusion_mode=${DIFFUSION_MODE},head_mask_warmup_frac=${HEAD_MASK_WARMUP_FRAC},progress_label=${progress_label}${model_args_extra}"

        ${env_prefix[@]} python -m accelerate.commands.launch --num_processes=1 eval_mask_head_llada.py \
            --model llada_mask_head_eval \
            --model_args "${common_args}" \
            --tasks "${task_name}" \
            "${include_path_args[@]}" \
            ${num_fewshot:+--num_fewshot ${num_fewshot}} \
            ${eval_batch_size:+--batch_size ${eval_batch_size}} \
            --limit ${LIMIT} \
            --output_path "${out_dir}/results.json" \
            --log_samples \
            $( [ "$task" = "humaneval" ] || [ "$task" = "mbpp" ] && echo "--confirm_run_unsafe_code" ) \
            2>&1 | tee "${out_dir}/eval.log"

        return ${PIPESTATUS[0]}
    }

    FAIL=0
    for task in "${TASKS[@]}"; do
        echo -e "\nRunning ${PRUNE_WHICH}: ${task}..."
        if run_single_eval "$task"; then
            echo "✅ Done: ${task}"
        else
            echo "❌ Failed: ${task}"
            FAIL=1
        fi
    done
done

exit $FAIL
