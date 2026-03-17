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

GEN_LENGTH=${GEN_LENGTH:-256}
STEPS=${STEPS:-256}
BLOCK_LENGTH=${BLOCK_LENGTH:-32}
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
    echo "HeadMask:    warmup_frac=${HEAD_MASK_WARMUP_FRAC}"
    echo "Out:         ${RUN_DIR}"
    echo "========================================================"

    run_single_eval() {
        local task=$1
        local out_dir="${RUN_DIR}/${task}"
        mkdir -p "$out_dir"

        local num_fewshot=""
        local eval_batch_size=""
        local model_args_extra=""
        local env_prefix=()

        case "$task" in
            mmlu|cmmlu|ceval-valid)
                num_fewshot=${MMLU_FEWSHOT:-5}
                eval_batch_size=1
                model_args_extra=",mc_num=1,cfg=0.0,is_check_greedy=False"
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

        local common_args="model_path=${MODEL_PATH}${importance_arg},prune_which=${PRUNE_WHICH},prune_k_frac=${PRUNE_K_FRAC},prune_scope=${PRUNE_SCOPE},random_prune_seed=${RANDOM_PRUNE_SEED},layer_start=${LAYER_START},layer_end=${LAYER_END},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},head_mask_warmup_frac=${HEAD_MASK_WARMUP_FRAC}${model_args_extra}"

        ${env_prefix[@]} python -m accelerate.commands.launch --num_processes=1 eval_mask_head_llada.py \
            --model llada_mask_head_eval \
            --model_args "${common_args}" \
            --tasks "${task}" \
            ${num_fewshot:+--num_fewshot ${num_fewshot}} \
            ${eval_batch_size:+--batch_size ${eval_batch_size}} \
            --limit ${LIMIT} \
            --output_path "${out_dir}/results.json" \
            --log_samples \
            $( [ "$task" = "humaneval" ] && echo "--confirm_run_unsafe_code" ) \
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
