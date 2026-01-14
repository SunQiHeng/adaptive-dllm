#!/bin/bash
# Head pruning / masking evaluation runner for LLaDA (lm-eval).
#
# 目标：
# - 根据 head importance 分数剪枝：剪最重要 top-k 或最不重要 top-k
# - 跑 lm-eval 任务（gsm8k / humaneval / mmlu 等）
#
# 用法示例：
#   PRUNE_WHICH=most PRUNE_K=64 TASKS_STR="mmlu" bash run_eval_mask_head_task.sh
#   PRUNE_WHICH=least PRUNE_K_FRAC=0.25 TASKS_STR="gsm8k,humaneval" bash run_eval_mask_head_task.sh
#
# 重要：
# - PRUNE_K 与 PRUNE_K_FRAC 二选一（必须提供一个）

set -o pipefail

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=/home/qiheng/Projects/adaptive-dllm:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2}

source ~/miniconda3/bin/activate adaptive-dllm

cd /home/qiheng/Projects/adaptive-dllm/evaluation/llada

# -----------------------
# Pruning config (Defaults)
# -----------------------
PRUNE_WHICH=${PRUNE_WHICH:-"most"}  # most|least|random

# -----------------------
# Model config
# -----------------------
MODEL_PATH=${MODEL_PATH:-"/data/qh_models/LLaDA-1.5"}
MODEL_NAME=${MODEL_NAME:-"llada_1_5"}

# importance file selection (复用 run_eval_task.sh 的映射逻辑的简化版)
MODEL_IMPORTANCE_ROOT=${MODEL_IMPORTANCE_ROOT:-""}
case "$MODEL_NAME" in
    llada-8b-instruct|LLaDA-8B-Instruct)
        MODEL_IMPORTANCE_ROOT="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada-8b-instruct_loss_gateIG"
        ;;
    llada-base|llada_base|llada|LLaDA-Base|LLaDA-base)
        MODEL_IMPORTANCE_ROOT="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base_loss_gateIG"
        ;;
    llada1.5|llada-1.5|llada-1_5|llada_1_5|LLaDA-1.5|LLaDA-1_5)
        MODEL_IMPORTANCE_ROOT="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada-1_5_loss_gateIG"
        ;;
    *)
        MODEL_IMPORTANCE_ROOT=""
        ;;
esac

IMPORTANCE_TAG=${IMPORTANCE_TAG:-"loss_gateIG_neg"}  # loss_gateIG | loss_gateIG_neg | all_ones
PRECOMPUTED_IMPORTANCE_PATH=""

# 如果走 random baseline，就不需要 importance 文件
if [ "$PRUNE_WHICH" != "random" ]; then
    if [ "$IMPORTANCE_TAG" = "all_ones" ]; then
        PRECOMPUTED_IMPORTANCE_PATH="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base_all_ones/head_importance.pt"
    elif [ "$IMPORTANCE_TAG" = "loss_gateIG" ]; then
        if [ -z "$MODEL_IMPORTANCE_ROOT" ]; then
            echo "ERROR: Unknown MODEL_NAME='${MODEL_NAME}' for IMPORTANCE_TAG=loss_gateIG. Please set MODEL_IMPORTANCE_ROOT."
            exit 2
        fi
        PRECOMPUTED_IMPORTANCE_PATH="${MODEL_IMPORTANCE_ROOT}/head_importance.pt"
    elif [ "$IMPORTANCE_TAG" = "loss_gateIG_neg" ]; then
        if [ -z "$MODEL_IMPORTANCE_ROOT" ]; then
            echo "ERROR: Unknown MODEL_NAME='${MODEL_NAME}' for IMPORTANCE_TAG=loss_gateIG_neg. Please set MODEL_IMPORTANCE_ROOT."
            exit 2
        fi
        SRC_IMPORTANCE_PATH="${MODEL_IMPORTANCE_ROOT}/head_importance.pt"
        NEG_DIR="${MODEL_IMPORTANCE_ROOT}_neg"
        PRECOMPUTED_IMPORTANCE_PATH="${NEG_DIR}/head_importance.pt"

        # Auto-generate negated importance if needed
        if [ ! -f "$PRECOMPUTED_IMPORTANCE_PATH" ]; then
            echo "➖ Generating negated importance..."
            python /home/qiheng/Projects/adaptive-dllm/evaluation/llada/generate_negated_importance.py \
                --in_pt "$SRC_IMPORTANCE_PATH" \
                --out_dir "$NEG_DIR"
            if [ ! -f "$PRECOMPUTED_IMPORTANCE_PATH" ]; then
                echo "ERROR: Failed to generate negated importance at: $PRECOMPUTED_IMPORTANCE_PATH"
                exit 3
            fi
        else
            echo "➖ Using existing negated importance: $PRECOMPUTED_IMPORTANCE_PATH"
        fi
    else
        echo "ERROR: Unknown IMPORTANCE_TAG='${IMPORTANCE_TAG}'."
        exit 2
    fi

    if [ ! -f "$PRECOMPUTED_IMPORTANCE_PATH" ]; then
        echo "ERROR: importance file not found: ${PRECOMPUTED_IMPORTANCE_PATH}"
        exit 3
    fi
fi

# -----------------------
# Pruning config (Details)
# -----------------------
PRUNE_K=${PRUNE_K:-""}
PRUNE_K_FRAC=${PRUNE_K_FRAC:-""}
LAYER_START=${LAYER_START:-0}
LAYER_END=${LAYER_END:-31}
RANDOM_PRUNE_SEED=${RANDOM_PRUNE_SEED:-1234}

# Default pruning amount:
# If user doesn't specify PRUNE_K / PRUNE_K_FRAC, default to pruning 25% heads per layer.
if [ -z "$PRUNE_K" ] && [ -z "$PRUNE_K_FRAC" ]; then
    PRUNE_K_FRAC="0.25"
    echo "⚠️  PRUNE_K / PRUNE_K_FRAC not set. Defaulting to PRUNE_K_FRAC=${PRUNE_K_FRAC} (prune 25% heads per layer)."
fi
if [ -n "$PRUNE_K" ] && [ -n "$PRUNE_K_FRAC" ]; then
    echo "ERROR: set only one of PRUNE_K or PRUNE_K_FRAC (not both)."
    exit 2
fi

# -----------------------
# Eval tasks / params
# -----------------------
TASKS=("gsm8k" "humaneval" "mmlu")
if [ -n "${TASKS_STR:-}" ]; then
    IFS=',' read -r -a TASKS <<< "${TASKS_STR}"
fi

GEN_LENGTH=${GEN_LENGTH:-256}
STEPS=${STEPS:-256}
BLOCK_LENGTH=${BLOCK_LENGTH:-32}
LIMIT=${LIMIT:-100}

RESULTS_ROOT="/home/qiheng/Projects/adaptive-dllm/evaluation/llada/${MODEL_NAME}_results/mask_head"
mkdir -p "$RESULTS_ROOT"

PRUNE_TAG=""
if [ -n "$PRUNE_K" ]; then
    PRUNE_TAG="prune_${PRUNE_WHICH}_k${PRUNE_K}"
else
    # sanitize float for path
    PRUNE_TAG="prune_${PRUNE_WHICH}_kfrac$(echo "${PRUNE_K_FRAC}" | tr '.' 'p')"
fi

RUN_TS=$(date +"%Y-%m-%dT%H-%M-%S")
RUN_DIR="${RESULTS_ROOT}/${IMPORTANCE_TAG}/${PRUNE_TAG}_L${LAYER_START}-${LAYER_END}_${RUN_TS}"
mkdir -p "$RUN_DIR"

echo "========================================================"
echo "Mask-Head / Pruning Eval"
echo "========================================================"
echo "Model:       ${MODEL_PATH}"
echo "Importance:  ${PRECOMPUTED_IMPORTANCE_PATH} (tag=${IMPORTANCE_TAG})"
echo "Prune:       which=${PRUNE_WHICH} k=${PRUNE_K:-"(none)"} k_frac=${PRUNE_K_FRAC:-"(none)"} layers=${LAYER_START}..${LAYER_END}"
echo "Tasks:       ${TASKS[*]}"
echo "Gen:         steps=${STEPS} gen_length=${GEN_LENGTH} block_length=${BLOCK_LENGTH}"
echo "Limit:       ${LIMIT}"
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
            # 加速：mc_num=1；其余沿用 eval_llada 的逻辑
            model_args_extra=",mc_num=1,cfg=0.0,is_check_greedy=False"
            if [ "${MMLU_OFFLINE:-1}" = "1" ]; then
                env_prefix=(env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1)
            fi
            ;;
        *)
            ;;
    esac

    local prune_args=""
    if [ -n "$PRUNE_K" ]; then
        prune_args=",prune_k=${PRUNE_K}"
    else
        prune_args=",prune_k_frac=${PRUNE_K_FRAC}"
    fi

    local importance_arg=""
    if [ "${PRUNE_WHICH}" != "random" ]; then
        # IMPORTANT:
        # Do NOT wrap in quotes here. lm-eval's model_args parser may keep quotes as literal characters,
        # resulting in a path like '"/abs/path.pt"' which breaks torch.load().
        importance_arg=",importance_path=${PRECOMPUTED_IMPORTANCE_PATH}"
    fi

    if [ "$task" = "humaneval" ]; then
        ${env_prefix[@]} python -m accelerate.commands.launch --num_processes=1 eval_mask_head_llada.py \
            --model llada_mask_head_eval \
            --model_args model_path="${MODEL_PATH}"${importance_arg},prune_which="${PRUNE_WHICH}"${prune_args},random_prune_seed=${RANDOM_PRUNE_SEED},layer_start=${LAYER_START},layer_end=${LAYER_END},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH}${model_args_extra} \
            --tasks "${task}" \
            ${num_fewshot:+--num_fewshot ${num_fewshot}} \
            ${eval_batch_size:+--batch_size ${eval_batch_size}} \
            --limit ${LIMIT} \
            --output_path "${out_dir}/results.json" \
            --log_samples \
            --confirm_run_unsafe_code \
            2>&1 | tee "${out_dir}/eval.log"
        return ${PIPESTATUS[0]}
    else
        ${env_prefix[@]} python -m accelerate.commands.launch --num_processes=1 eval_mask_head_llada.py \
            --model llada_mask_head_eval \
            --model_args model_path="${MODEL_PATH}"${importance_arg},prune_which="${PRUNE_WHICH}"${prune_args},random_prune_seed=${RANDOM_PRUNE_SEED},layer_start=${LAYER_START},layer_end=${LAYER_END},gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH}${model_args_extra} \
            --tasks "${task}" \
            ${num_fewshot:+--num_fewshot ${num_fewshot}} \
            ${eval_batch_size:+--batch_size ${eval_batch_size}} \
            --limit ${LIMIT} \
            --output_path "${out_dir}/results.json" \
            --log_samples \
            2>&1 | tee "${out_dir}/eval.log"
        return ${PIPESTATUS[0]}
    fi
}

FAIL=0
for task in "${TASKS[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Running: ${task}"
    echo "----------------------------------------"
    if run_single_eval "$task"; then
        echo "✅ Done: ${task}"
    else
        echo "❌ Failed: ${task} (see ${RUN_DIR}/${task}/eval.log)"
        FAIL=1
    fi
done

exit $FAIL

