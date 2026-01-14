#!/bin/bash
# Dream head pruning / masking evaluation runner (lm-eval).
#
# 支持三种剪枝模式（全局）：
# - PRUNE_WHICH=most   : 全局剪掉最重要 top-k heads（依赖 importance_path）
# - PRUNE_WHICH=least  : 全局剪掉最不重要 top-k heads（依赖 importance_path）
# - PRUNE_WHICH=random : 全局随机剪枝 top-k heads（不依赖 importance_path）
#
# PRUNE_K / PRUNE_K_FRAC 二选一；如果都不传，默认 PRUNE_K_FRAC=0.25。

set -o pipefail

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=/home/qiheng/Projects/adaptive-dllm:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5}

source ~/miniconda3/bin/activate adaptive-dllm

cd /home/qiheng/Projects/adaptive-dllm/evaluation/dream

# -----------------------
# Pruning config (Defaults)
# -----------------------
PRUNE_WHICH=${PRUNE_WHICH:-"random"}  # most|least|random
PRUNE_K=${PRUNE_K:-""}
PRUNE_K_FRAC=${PRUNE_K_FRAC:-""}
LAYER_START=${LAYER_START:-0}
LAYER_END=${LAYER_END:-31}
RANDOM_PRUNE_SEED=${RANDOM_PRUNE_SEED:-1234}

if [ -z "$PRUNE_K" ] && [ -z "$PRUNE_K_FRAC" ]; then
    PRUNE_K_FRAC="0.25"
    echo "⚠️  PRUNE_K / PRUNE_K_FRAC not set. Defaulting to PRUNE_K_FRAC=${PRUNE_K_FRAC} (global prune 25% heads)."
fi
if [ -n "$PRUNE_K" ] && [ -n "$PRUNE_K_FRAC" ]; then
    echo "ERROR: set only one of PRUNE_K or PRUNE_K_FRAC (not both)."
    exit 2
fi

# -----------------------
# Model / importance config
# -----------------------
MODEL_PATH=${MODEL_PATH:-"/data/qh_models/Dream-v0-Instruct-7B"}
MODEL_NAME=${MODEL_NAME:-"dream"}

IMPORTANCE_TAG=${IMPORTANCE_TAG:-"loss_gateIG"}  # loss_gateIG | loss_gateIG_neg | loss_gateIG_zero | loss_gateIG_zero_neg | all_ones
PRECOMPUTED_IMPORTANCE_PATH=""

if [ "$PRUNE_WHICH" != "random" ]; then
    if [ "$IMPORTANCE_TAG" = "loss_gateIG" ]; then
        PRECOMPUTED_IMPORTANCE_PATH="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream_loss_gateIG/head_importance.pt"
    elif [ "$IMPORTANCE_TAG" = "loss_gateIG_neg" ]; then
        SRC_IMPORTANCE_PATH="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream_loss_gateIG/head_importance.pt"
        NEG_DIR="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream_loss_gateIG_neg"
        PRECOMPUTED_IMPORTANCE_PATH="${NEG_DIR}/head_importance.pt"
        if [ ! -f "$PRECOMPUTED_IMPORTANCE_PATH" ]; then
            echo "➖ Generating negated importance..."
            python /home/qiheng/Projects/adaptive-dllm/evaluation/dream/generate_negated_importance.py \
                --in_pt "$SRC_IMPORTANCE_PATH" \
                --out_dir "$NEG_DIR"
        fi
    elif [ "$IMPORTANCE_TAG" = "loss_gateIG_zero" ]; then
        PRECOMPUTED_IMPORTANCE_PATH="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream_base_loss_gateIG_zero_maskp0.15-0.3-0.5-0.7-0.9_mcs2_mean_masked_seed47_n50_k8_L2048_dseed47_mseed47_ts20251227_191418/head_importance.pt"
    elif [ "$IMPORTANCE_TAG" = "loss_gateIG_zero_neg" ]; then
        SRC_IMPORTANCE_PATH="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream_base_loss_gateIG_zero_maskp0.15-0.3-0.5-0.7-0.9_mcs2_mean_masked_seed47_n50_k8_L2048_dseed47_mseed47_ts20251227_191418/head_importance.pt"
        NEG_DIR="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream_base_loss_gateIG_zero_neg"
        PRECOMPUTED_IMPORTANCE_PATH="${NEG_DIR}/head_importance.pt"
        if [ ! -f "$PRECOMPUTED_IMPORTANCE_PATH" ]; then
            echo "➖ Generating negated importance..."
            python /home/qiheng/Projects/adaptive-dllm/evaluation/dream/generate_negated_importance.py \
                --in_pt "$SRC_IMPORTANCE_PATH" \
                --out_dir "$NEG_DIR"
        fi
    elif [ "$IMPORTANCE_TAG" = "all_ones" ]; then
        PRECOMPUTED_IMPORTANCE_PATH="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream_base_all_ones/head_importance.pt"
    else
        echo "ERROR: Unknown IMPORTANCE_TAG='$IMPORTANCE_TAG'."
        exit 2
    fi

    if [ ! -f "$PRECOMPUTED_IMPORTANCE_PATH" ]; then
        echo "ERROR: importance file not found: ${PRECOMPUTED_IMPORTANCE_PATH}"
        exit 3
    fi
fi

# -----------------------
# Eval params
# -----------------------
TASKS=("gsm8k" "humaneval" "mmlu")
if [ -n "${TASKS_STR:-}" ]; then
    IFS=',' read -r -a TASKS <<< "${TASKS_STR}"
fi

# Official-ish Dream params (match eval_dream.sh defaults)
TEMPERATURE=${TEMPERATURE:-0.1}
TOP_P=${TOP_P:-0.9}
ALG=${ALG:-"entropy"}
ALG_TEMP=${ALG_TEMP:-0.0}
BLOCK_SIZE=${BLOCK_SIZE:-32}
LIMIT=${LIMIT:-100}

RESULTS_ROOT="/home/qiheng/Projects/adaptive-dllm/evaluation/dream/results/mask_head"
mkdir -p "$RESULTS_ROOT"

PRUNE_TAG=""
if [ -n "$PRUNE_K" ]; then
    PRUNE_TAG="prune_${PRUNE_WHICH}_k${PRUNE_K}"
else
    PRUNE_TAG="prune_${PRUNE_WHICH}_kfrac$(echo "${PRUNE_K_FRAC}" | tr '.' 'p')"
fi

RUN_TS=$(date +"%Y-%m-%dT%H-%M-%S")
RUN_DIR="${RESULTS_ROOT}/${IMPORTANCE_TAG}/${PRUNE_TAG}_L${LAYER_START}-${LAYER_END}_${RUN_TS}"
mkdir -p "$RUN_DIR"

echo "========================================================"
echo "Dream Mask-Head / Pruning Eval"
echo "========================================================"
echo "Model:      ${MODEL_PATH}"
echo "Prune:      which=${PRUNE_WHICH} k=${PRUNE_K:-"(none)"} k_frac=${PRUNE_K_FRAC:-"(none)"} layers=${LAYER_START}..${LAYER_END}"
echo "Seed:       ${RANDOM_PRUNE_SEED} (random mode only)"
echo "Importance: ${PRECOMPUTED_IMPORTANCE_PATH} (tag=${IMPORTANCE_TAG})"
echo "Tasks:      ${TASKS[*]}"
echo "Out:        ${RUN_DIR}"
echo "========================================================"

run_single_eval() {
    local task=$1
    local out_dir="${RUN_DIR}/${task}"
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
    fi
    if [ "$task" = "mmlu" ]; then
        # eval_dream.py MMLU 路径通常走 apply_chat_template + batch_size 1
        apply_chat="--apply_chat_template"
        num_fewshot=${MMLU_FEWSHOT:-5}
        eval_batch_size=1
        model_args_extra=",mc_num=1,likelihood_now_step=${steps},recompute_mask_each_call=true"
        if [ "${MMLU_OFFLINE:-1}" = "1" ]; then
            env_prefix=(env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1)
        fi
    fi

    local prune_args=""
    if [ -n "$PRUNE_K" ]; then
        prune_args=",prune_k=${PRUNE_K}"
    else
        prune_args=",prune_k_frac=${PRUNE_K_FRAC}"
    fi

    local importance_arg=""
    if [ "$PRUNE_WHICH" != "random" ]; then
        # 不要额外加引号（lm-eval parser 会把引号当字面量）
        importance_arg=",importance_path=${PRECOMPUTED_IMPORTANCE_PATH}"
    fi

    local model_args="model_path=${MODEL_PATH},max_new_tokens=${max_new_tokens},steps=${steps},temperature=${TEMPERATURE},top_p=${TOP_P},alg=${ALG},alg_temp=${ALG_TEMP},block_size=${BLOCK_SIZE}${model_args_extra}${importance_arg},prune_which=${PRUNE_WHICH}${prune_args},random_prune_seed=${RANDOM_PRUNE_SEED},layer_start=${LAYER_START},layer_end=${LAYER_END}"

    if [ "$task" = "humaneval" ]; then
        ${env_prefix[@]} python -m accelerate.commands.launch --num_processes=1 eval_mask_head_dream.py \
            --model dream_mask_head_eval \
            --model_args "${model_args}" \
            --tasks "${task}" \
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
            --tasks "${task}" \
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

