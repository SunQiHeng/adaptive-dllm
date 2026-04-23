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
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}

source ~/miniconda3/bin/activate adaptive-dllm

cd /home/qiheng/Projects/adaptive-dllm/evaluation/llada

# -----------------------
# Model config
# -----------------------
MODEL_PATH=${MODEL_PATH:-"/data/qh_models/LLaDA-1.5"}
# Result directory label for the evaluated model.
EVAL_MODEL_NAME=${EVAL_MODEL_NAME:-"llada_1_5"}

# -----------------------
# Pruning / Importance config
# -----------------------
# 现在支持一次运行多个模式，用逗号分隔，例如: MODES="most,least,random"
MODES_STR=${MODES:-"most,least,random"}
IFS=',' read -r -a MODES <<< "${MODES_STR}"

PRUNE_SCOPE=${PRUNE_SCOPE:-"layer"} # global|layer (全局排序剪枝 vs 每层按相同比例剪枝)
# Preferred interface:
#   Single dataset: MODEL_NAME=dream ATTR_METHOD=headig ATTR_DATASETS_STR=ceval-valid_all
#   Multi dataset:  MODEL_NAME=dream ATTR_METHOD=headig ATTR_DATASETS_STR="mmlu_all,cmmlu_all"
# You can still override with IMPORTANCE_PATH directly.
ATTR_MODEL_NAME=${MODEL_NAME:-"llada-1_5"}
# ATTR_METHOD candidates:
#   headig | attnlrp | shapley | attarr
ATTR_METHOD=${ATTR_METHOD:-"headig"}
# ATTR_DATASETS_STR candidates:
#   mmlu_all | cmmlu_all | ceval-valid_all | gsm8k | minerva_math | gpqa_main_n_shot_all | humaneval | mbpp
# ATTR_DATASETS_STR=${ATTR_DATASETS_STR:-"mmlu_all,cmmlu_all,ceval-valid_all,gsm8k,gpqa_main_n_shot_all,humaneval,mbpp"}
ATTR_DATASETS_STR=${ATTR_DATASETS_STR:-"gsm8k,gpqa_main_n_shot_all,humaneval,mbpp"}
IFS=',' read -r -a ATTR_DATASETS <<< "${ATTR_DATASETS_STR}"
FIRST_ATTR_DATASET="$(echo "${ATTR_DATASETS[0]}" | xargs)"
USER_IMPORTANCE_PATH=${IMPORTANCE_PATH:-""}
build_default_importance_path() {
    local attr_dataset="$1"
    echo "/home/qiheng/Projects/adaptive-dllm/configs/${ATTR_MODEL_NAME}_${ATTR_METHOD}_${attr_dataset}/head_importance.pt"
}

build_aconfig_importance_path() {
    local attr_dataset="$1"
    python - "$ATTR_MODEL_NAME" "$ATTR_METHOD" "$attr_dataset" <<'PY'
from pathlib import Path
import sys

model_name = sys.argv[1]
attr_method = sys.argv[2]
attr_dataset = sys.argv[3]
root = Path("/home/qiheng/Projects/adaptive-dllm/configs/aconfigs")
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
# Recommended LIMIT values for task-specific evals (stability vs runtime trade-off):
#   mmlu_all: 40
#   cmmlu_all: 40
#   ceval-valid_all: 200
#   gpqa_main_n_shot_all: 200
#   gsm8k: 200
#   minerva_math: 200
#   humaneval: 200
#   mbpp: 200
# Negation modes:
#   USE_NEGATED_MODES_STR="0"   -> only original importance
#   USE_NEGATED_MODES_STR="1"   -> only negated importance
#   USE_NEGATED_MODES_STR="0,1" -> run both original and negated importance
USE_NEGATED_MODES_STR=${USE_NEGATED_MODES_STR:-"0"}
IFS=',' read -r -a USE_NEGATED_MODES <<< "${USE_NEGATED_MODES_STR}"
IMPORTANCE_TAG_OVERRIDE=${IMPORTANCE_TAG:-""}

USED_IMPORTANCE_PATH="${IMPORTANCE_PATH}"
IMPORTANCE_TAG=""

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

PRUNE_K_FRAC=${PRUNE_K_FRAC:-"0.2"}
LAYER_START=${LAYER_START:-0}
LAYER_END=${LAYER_END:-31}
RANDOM_PRUNE_SEED=${RANDOM_PRUNE_SEED:-1234}
HEAD_MASK_WARMUP_FRAC=${HEAD_MASK_WARMUP_FRAC:-0.2}

# -----------------------
# Eval tasks / params
# -----------------------
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

default_limit_for_attr_dataset() {
    case "$1" in
        mmlu_all) echo 40 ;;
        cmmlu_all) echo 40 ;;
        ceval-valid_all) echo 200 ;;
        gpqa_main_n_shot_all) echo 200 ;;
        gsm8k) echo 200 ;;
        minerva_math) echo 200 ;;
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
LOCAL_TASK_ROOT="/home/qiheng/Projects/adaptive-dllm/evaluation/local_tasks/generated"
DEFAULT_GPQA_DATA_PATH="/home/qiheng/Projects/adaptive-dllm/evaluation/local_data/gpqa/gpqa_main.jsonl"
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

NUM_FEWSHOT=${NUM_FEWSHOT:-0}

# -----------------------
# Run Loop
# -----------------------
# -----------------------
echo "Attr datasets: ${ATTR_DATASETS[*]}"
if [ -n "${USER_IMPORTANCE_PATH}" ]; then
    echo "Importance base: ${USER_IMPORTANCE_PATH} (manual override for all datasets)"
else
    echo "Importance base: auto-resolved per item in ATTR_DATASETS_STR"
fi

TASKS_PER_DATASET=${#TASKS[@]}
TOTAL_TASKS=$((${#ATTR_DATASETS[@]} * ${#USE_NEGATED_MODES[@]} * ${#MODES[@]} * ${TASKS_PER_DATASET}))
CURRENT_TASK=0

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

        for current_mode in "${MODES[@]}"; do
            PRUNE_WHICH=$current_mode
            
            # -----------------------
            # Output config
            # -----------------------
            RESULTS_ROOT="/home/qiheng/Projects/adaptive-dllm/evaluation/llada/${EVAL_MODEL_NAME}_results/mask_head"
            PRUNE_TAG="prune_${PRUNE_WHICH}_scope_${PRUNE_SCOPE}_kfrac$(echo "${PRUNE_K_FRAC}" | tr '.' 'p')"
            RUN_TS=$(date +"%Y-%m-%dT%H-%M-%S")
            RUN_DIR="${RESULTS_ROOT}/${IMPORTANCE_TAG}/${PRUNE_TAG}_L${LAYER_START}-${LAYER_END}_${RUN_TS}"
            mkdir -p "$RUN_DIR"

    run_single_eval() {
        local task=$1
        local out_dir="${RUN_DIR}/${task}"
        local task_name="${task}"
        local progress_label="${task}|mask_head|$(basename "${RUN_DIR}")"
        mkdir -p "$out_dir"

        local local_gen_length=${GEN_LENGTH}
        local local_steps=${STEPS}
        case "$task" in
            humaneval)
                local_gen_length=${HUMANEVAL_GEN_LENGTH:-768}
                local_steps=${HUMANEVAL_STEPS:-${local_gen_length}}
                ;;
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
                if [ ! -f "$USED_IMPORTANCE_PATH" ]; then
                    echo "ERROR: importance file not found: ${USED_IMPORTANCE_PATH}"
                exit 3
            fi
                importance_arg=",importance_path=${USED_IMPORTANCE_PATH}"
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
                CURRENT_TASK=$((CURRENT_TASK + 1))
                CURRENT_RUN_LABEL="[${CURRENT_TASK}/${TOTAL_TASKS}] dataset=${CURRENT_ATTR_DATASET} variant=${negation_label} prune=${PRUNE_WHICH} task=${task}"
                echo "[start] ${CURRENT_RUN_LABEL} limit=${LIMIT} out=${RUN_DIR}/${task}"
                if run_single_eval "$task"; then
                    echo "[done]  ${CURRENT_RUN_LABEL}"
                else
                    echo "[fail]  ${CURRENT_RUN_LABEL} log=${RUN_DIR}/${task}/eval.log"
                    FAIL=1
                fi
            done
        done
    done
done

exit $FAIL
