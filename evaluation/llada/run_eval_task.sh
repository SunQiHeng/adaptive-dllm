#!/bin/bash
# Quick test script for GSM8K and HumanEval
# Tests 3 model types on 2 tasks with reduced parameters
# Usage: bash run_eval_quick_test.sh

# Make pipelines fail if the left-hand command fails (e.g., when piping to tee).
set -o pipefail

# Environment setup
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=/home/qiheng/Projects/adaptive-dllm:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1

# Activate environment
source ~/miniconda3/bin/activate adaptive-dllm

cd /home/qiheng/Projects/adaptive-dllm/evaluation/llada

# Model configuration
MODEL_PATH=${MODEL_PATH:-"/data/qh_models/LLaDA-1.5"}
MODEL_NAME=${MODEL_NAME:-"llada_1_5"}
# Model types to run (can be overridden without editing file):
#   MODEL_TYPES_STR="standard,sparse,adaptive" bash run_eval_task.sh
MODEL_TYPES=("adaptive")
if [ -n "${MODEL_TYPES_STR:-}" ]; then
    IFS=',' read -r -a MODEL_TYPES <<< "${MODEL_TYPES_STR}"
fi

# Output root (all results go here)
RESULTS_ROOT="/home/qiheng/Projects/adaptive-dllm/evaluation/llada/${MODEL_NAME}_results"
mkdir -p "$RESULTS_ROOT"

# Model-specific head-importance (used for loss_gateIG variants)
#
# You can extend this mapping if you add more models.
# You may also override it directly via env: MODEL_IMPORTANCE_ROOT=/path/to/config_dir
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

# Which precomputed head-importance to use for adaptive mode:
# - margin:       /home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base_margin/head_importance.pt
# - target_logit: /home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base_target_logit/head_importance.pt
#
# Shuffle ablations (distribution preserved, head indices permuted within each layer):
# - margin_shuf
# - target_logit_shuf
#
IMPORTANCE_TAG=${IMPORTANCE_TAG:-"loss_gateIG"}  # margin | target_logit | all_ones | margin_shuf | target_logit_shuf | loss_gateIG | loss_gateIG_neg
SHUFFLE_SEED=${SHUFFLE_SEED:-1234}
if [ "$IMPORTANCE_TAG" = "margin" ]; then
    PRECOMPUTED_IMPORTANCE_PATH="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base_margin/head_importance.pt"
elif [ "$IMPORTANCE_TAG" = "target_logit" ]; then
    PRECOMPUTED_IMPORTANCE_PATH="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base_target_logit/head_importance.pt"
elif [ "$IMPORTANCE_TAG" = "all_ones" ]; then
    PRECOMPUTED_IMPORTANCE_PATH="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base_all_ones/head_importance.pt"
elif [ "$IMPORTANCE_TAG" = "margin_shuf" ]; then
    SRC_IMPORTANCE_PATH="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base_margin/head_importance.pt"
    SHUF_DIR="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base_margin_shuf_seed${SHUFFLE_SEED}"
    PRECOMPUTED_IMPORTANCE_PATH="${SHUF_DIR}/head_importance.pt"
elif [ "$IMPORTANCE_TAG" = "target_logit_shuf" ]; then
    SRC_IMPORTANCE_PATH="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base_target_logit/head_importance.pt"
    SHUF_DIR="/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base_target_logit_shuf_seed${SHUFFLE_SEED}"
    PRECOMPUTED_IMPORTANCE_PATH="${SHUF_DIR}/head_importance.pt"
elif [ "$IMPORTANCE_TAG" = "loss_gateIG" ]; then
    if [ -z "$MODEL_IMPORTANCE_ROOT" ]; then
        echo "ERROR: Unknown MODEL_NAME='${MODEL_NAME}' for IMPORTANCE_TAG=loss_gateIG."
        echo "Please set MODEL_NAME to one of: llada-8b-instruct, llada-1_5 (or extend the mapping in this script)."
        exit 2
    fi
    PRECOMPUTED_IMPORTANCE_PATH="${MODEL_IMPORTANCE_ROOT}/head_importance.pt"
elif [ "$IMPORTANCE_TAG" = "loss_gateIG_neg" ]; then
    if [ -z "$MODEL_IMPORTANCE_ROOT" ]; then
        echo "ERROR: Unknown MODEL_NAME='${MODEL_NAME}' for IMPORTANCE_TAG=loss_gateIG_neg."
        echo "Please set MODEL_NAME to one of: llada-8b-instruct, llada-1_5 (or extend the mapping in this script)."
        exit 2
    fi
    SRC_IMPORTANCE_PATH="${MODEL_IMPORTANCE_ROOT}/head_importance.pt"
    NEG_DIR="${MODEL_IMPORTANCE_ROOT}_neg"
    PRECOMPUTED_IMPORTANCE_PATH="${NEG_DIR}/head_importance.pt"
else
    echo "ERROR: Unknown IMPORTANCE_TAG='$IMPORTANCE_TAG'. Use 'margin', 'target_logit', 'all_ones', 'margin_shuf', 'target_logit_shuf', 'loss_gateIG', or 'loss_gateIG_neg'."
    exit 2
fi

# Auto-generate shuffled importance if needed
if [ "$IMPORTANCE_TAG" = "margin_shuf" ] || [ "$IMPORTANCE_TAG" = "target_logit_shuf" ]; then
    if [ ! -f "$PRECOMPUTED_IMPORTANCE_PATH" ]; then
        echo "🔀 Generating shuffled importance (seed=${SHUFFLE_SEED})..."
        python /home/qiheng/Projects/adaptive-dllm/evaluation/llada/generate_shuffled_importance.py \
            --in_pt "$SRC_IMPORTANCE_PATH" \
            --out_dir "$SHUF_DIR" \
            --seed "$SHUFFLE_SEED"
        if [ ! -f "$PRECOMPUTED_IMPORTANCE_PATH" ]; then
            echo "ERROR: Failed to generate shuffled importance at: $PRECOMPUTED_IMPORTANCE_PATH"
            exit 3
        fi
    else
        echo "🔀 Using existing shuffled importance: $PRECOMPUTED_IMPORTANCE_PATH"
    fi
fi

# Auto-generate negated importance if needed
if [ "$IMPORTANCE_TAG" = "loss_gateIG_neg" ]; then
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
fi

# Generation parameters
GEN_LENGTH=256
STEPS=256
BLOCK_LENGTH=32
BLOCK_SIZE=32
LIMIT=100

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

# Default to local exported JSONL if present
if [ -z "$RULER_DATA_PATH" ] && [ -d "/data/qh_models/ruler/jsonl/${RULER_SPLIT}" ]; then
    RULER_DATA_PATH="/data/qh_models/ruler/jsonl/${RULER_SPLIT}"
fi

# Tasks to run (can be overridden without editing file):
#   TASKS_STR="mmlu,ruler" bash run_eval_task.sh
TASKS=("gsm8k" "humaneval" "mmlu" "ruler")
if [ -n "${TASKS_STR:-}" ]; then
    IFS=',' read -r -a TASKS <<< "${TASKS_STR}"
fi

echo "========================================================"
echo "Quick Test Configuration"
echo "========================================================"
echo "Tasks: ${TASKS[*]}"
echo "Model Types: standard, sparse, adaptive"
echo "Gen Length: ${GEN_LENGTH}, Steps: ${STEPS}, Block Length: ${BLOCK_LENGTH}, Block Size: ${BLOCK_SIZE}"
echo "Limit: ${LIMIT}"
echo "RULER len_k (if enabled): ${RULER_LEN_K}k"
echo "Importance tag: ${IMPORTANCE_TAG}"
echo "Importance file: ${PRECOMPUTED_IMPORTANCE_PATH}"
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
    
    # Record start time
    START_TIME=$(date +%s)
    
    echo "Params: gen_length=${GEN_LENGTH}, steps=${STEPS}, block_length=${BLOCK_LENGTH}, block_size=${BLOCK_SIZE}, limit=${LIMIT}"
    
    # Set importance source for adaptive mode
    if [ "$model_type" = "adaptive" ]; then
        IMPORTANCE_ARG=",importance_source=precomputed,precomputed_importance_path=${PRECOMPUTED_IMPORTANCE_PATH}"
    else
        IMPORTANCE_ARG=""
    fi

    # Task-specific settings
    #
    # Notes:
    # - Previously this script did not pass --num_fewshot, so tasks used lm-eval defaults (often 0-shot).
    # - For multiple-choice likelihood tasks like MMLU, LLaDAEvalHarness uses internal MC sampling (mc_num/batch_size).
    #   Setting mc_num=1 and using lm-eval's --batch_size 1 makes MMLU runs much faster and matches eval_llada.sh usage.
    NUM_FEWSHOT=""
    MODEL_ARGS_EXTRA=""
    EVAL_BATCH_SIZE=""
    ENV_PREFIX=()
    case "$task" in
        mmlu|cmmlu|ceval-valid)
            NUM_FEWSHOT=${MMLU_FEWSHOT:-5}
            # IMPORTANT: don't pass batch_size via --model_args; lm-eval already passes batch_size to the model ctor.
            # Passing it twice triggers: "got multiple values for keyword argument 'batch_size'".
            EVAL_BATCH_SIZE=1
            # MMLU uses loglikelihood scoring; enable sparse/adaptive effects by setting now_step > warmup
            # and recomputing masks each forward (masks depend on content).
            MODEL_ARGS_EXTRA=",mc_num=1,cfg=0.0,is_check_greedy=False,likelihood_now_step=${STEPS},recompute_mask_each_call=true"
            # Make MMLU robust to HF Hub flakiness (502) by default: use cached datasets only.
            # NOTE: we must use `env` here; `FOO=bar` produced by variable expansion is NOT treated as assignment by bash.
            # You can disable this via: MMLU_OFFLINE=0 bash run_eval_task.sh
            if [ "${MMLU_OFFLINE:-1}" = "1" ]; then
                ENV_PREFIX=(env HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1)
            fi
            ;;
        ruler)
            # Handled by eval_ruler_llada.py below (not lm-eval tasks)
            ;;
        *)
            # Keep lm-eval defaults unless you extend this case block.
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
            --skip 0.2 \
            --select 0.3 \
            --block_size "${BLOCK_SIZE}" \
            --importance_source precomputed \
            --precomputed_importance_path "${PRECOMPUTED_IMPORTANCE_PATH}" \
            --len_k "${RULER_LEN_K}" \
            --limit "${RULER_LIMIT}" \
            --output_path "${OUTPUT_DIR}/results.json" \
            --samples_path "${OUTPUT_DIR}/samples.jsonl" \
            "${RULER_ARGS[@]}" \
            2>&1 | tee "${OUTPUT_DIR}/eval.log"
        CMD_RC=${PIPESTATUS[0]}
    elif [ "$task" = "humaneval" ]; then
        # HumanEval requires --confirm_run_unsafe_code
        ${ENV_PREFIX[@]} python -m accelerate.commands.launch --num_processes=1 eval_llada.py \
            --model llada_eval \
            --model_args model_path="${MODEL_PATH}",model_type="${model_type}",gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},skip=0.2,select=0.3,block_size=${BLOCK_SIZE}${IMPORTANCE_ARG}${MODEL_ARGS_EXTRA} \
            --tasks "${task}" \
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
            --model_args model_path="${MODEL_PATH}",model_type="${model_type}",gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},skip=0.2,select=0.3,block_size=${BLOCK_SIZE}${IMPORTANCE_ARG}${MODEL_ARGS_EXTRA} \
            --tasks "${task}" \
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
    echo "Task: ${task}"
    for model_type in "${MODEL_TYPES[@]}"; do
        RESULT_FILE="${RESULTS_ROOT}/${model_type}/${task}_${IMPORTANCE_TAG}/results.json"
        if [ -f "$RESULT_FILE" ]; then
            echo "  ✅ ${model_type}: ${RESULTS_ROOT}/${model_type}/${task}_${IMPORTANCE_TAG}/"
        else
            echo "  ❌ ${model_type}: FAILED"
        fi
    done
    echo ""
done

