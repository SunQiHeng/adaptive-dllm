#!/usr/bin/env bash
set -euo pipefail

# One-command runner for loss-based head attribution (ALL-HEADS JOINT IG).
# Default behavior:
#   - run all 8 unified tasks serially on one GPU
#   - use per-dataset default attribution sample counts
# Usage:
#   bash /home/qiheng/Projects/adaptive-dllm/models/LLaDA/attribution/loss_attribution/run_loss_attribution_all_heads.sh
#
# Useful overrides (env vars):
#   ATTR_DATASETS_STR="mmlu,gsm8k,humaneval"
#   MAX_SAMPLES=80
#   MMLU_MAX_SAMPLES=200 GSM8K_MAX_SAMPLES=100 HUMANEVAL_MAX_SAMPLES=100
#   GPQA_DATA_PATH=/path/to/gpqa_main.jsonl

mkdir -p logs

echo "========================================================"
echo "LLaDA Loss Attribution (All-heads Joint IG)"
echo "========================================================"
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-"(unset)"}"
echo "========================================================"

# Pin to a specific GPU id (default: 4; keep consistent with your existing runner)
GPU_ID=${GPU_ID:-0}
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "Pinned GPU via CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Optional: activate conda env (non-fatal if not present)
if [ -f "$HOME/miniconda3/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/bin/activate" adaptive-dllm || true
fi

export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=/home/qiheng/Projects/adaptive-dllm:${PYTHONPATH:-}

MODEL_PATH=${MODEL_PATH:-"/data/qh_models/LLaDA-1.5"}
MODEL_NAME=${MODEL_NAME:-"llada-1_5"}
DEFAULT_GPQA_DATA_PATH="/home/qiheng/Projects/adaptive-dllm/evaluation/local_data/gpqa/gpqa_main.jsonl"

# Where to write results
OUT_ROOT=${OUT_ROOT:-"/home/qiheng/Projects/adaptive-dllm/configs"}
RUN_TS=${RUN_TS:-"$(date +%Y%m%d_%H%M%S)"}

# Attribution dataset (can differ from downstream eval tasks)
ATTR_DATASET=${ATTR_DATASET:-"mmlu"}     # nemotron | gsm8k | minerva_math | mmlu | cmmlu | ceval-valid | gpqa_main_n_shot | humaneval | mbpp
ATTR_DATASETS_STR=${ATTR_DATASETS_STR:-"mmlu,cmmlu,ceval-valid,gsm8k,minerva_math,gpqa_main_n_shot,humaneval,mbpp"}
SPLIT=${SPLIT:-"test"}                      # dataset split name (gsm8k/mmlu); humaneval uses fixed test
DATA_PATH=${DATA_PATH:-""}
SAMPLES_PER_CATEGORY=${SAMPLES_PER_CATEGORY:-10}  # nemotron only
NEMOTRON_CATEGORIES=${NEMOTRON_CATEGORIES:-"code,math,science,chat,safety"} # nemotron only
MMLU_SUBJECT=${MMLU_SUBJECT:-"all"}        # mmlu only (e.g. abstract_algebra, anatomy, ... or 'all')
BASE_DATA_PATH=${DATA_PATH:-""}
GLOBAL_MAX_SAMPLES=${MAX_SAMPLES:-""}
IG_STEPS=${IG_STEPS:-8}
MAX_LENGTH=${MAX_LENGTH:-2048}
MIN_COMPLETION_TOKENS=${MIN_COMPLETION_TOKENS:-256}   # nemotron recommended; 0 disables
SEED=${SEED:-123}

# Seeds:
# - DATA_SEED controls which samples are selected (dataset subsampling/shuffle).
# - MASK_SEED controls random masking positions (diffusion-style masking).
DATA_SEED=${DATA_SEED:-${SEED}}
MASK_SEED=${MASK_SEED:-${SEED}}

DATASET_SHUFFLE=${DATASET_SHUFFLE:-1}  # 1 => enable --dataset_shuffle, 0 => disable

NEMOTRON_POOL_PER_CATEGORY=${NEMOTRON_POOL_PER_CATEGORY:-3000}

# IG baseline:
BASELINE=${BASELINE:-"zero"}        # zero | scalar
BASELINE_SCALAR=${BASELINE_SCALAR:-0.3}

# Multi-timestep diffusion-style masking
MASK_PROBS=${MASK_PROBS:-"0.15,0.3,0.5,0.7,0.9"}
MASK_SAMPLES_PER_PROB=${MASK_SAMPLES_PER_PROB:-2}
LOSS_NORMALIZE=${LOSS_NORMALIZE:-"mean_masked"}  # mean_masked | sum
IG_POSTPROCESS=${IG_POSTPROCESS:-"signed"}       # abs | signed | relu
MASK_BATCH_SIZE=${MASK_BATCH_SIZE:-2}            # 0 => all variants in one batch (may OOM)
ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING:-"whole_layer"}  # none | whole_layer | one_in_two | ...

# Debug (optional)
DEBUG_DUMP_SAMPLES=${DEBUG_DUMP_SAMPLES:-10}          # e.g. 10 => print first 10 sample fingerprints
DEBUG_SAVE_PER_SAMPLE=${DEBUG_SAVE_PER_SAMPLE:-1}    # e.g. 8 => save per_sample_ig.pt for first 8 processed samples
SHOW_PROGRESS=${SHOW_PROGRESS:-1}
PROGRESS_UPDATE_EVERY=${PROGRESS_UPDATE_EVERY:-10}

PROGRESS_FLAG="--no_progress"
if [ "${SHOW_PROGRESS}" = "1" ]; then
  PROGRESS_FLAG="--progress_update_every ${PROGRESS_UPDATE_EVERY}"
fi

# Path mode (design fix for attribution similarity)
PATH_MODE=${PATH_MODE:-"random_threshold"}   # random_threshold | diagonal
PATH_SAMPLES=${PATH_SAMPLES:-4}             # only used when PATH_MODE=random_threshold
PATH_SEED=${PATH_SEED:--1}                  # -1 => use mask_seed

# Layer range (inclusive). -1 means last layer.
LAYER_START=${LAYER_START:-0}
LAYER_END=${LAYER_END:-31}

# GSM8K attribution target:
# - final:      supervise only final answer tokens (after '####')
# - final_hash: supervise "#### <final>" (closer to lm-eval extraction pattern)
# - full:       supervise full `answer` field (rationale + final), usually more stable
GSM8K_ANSWER_MODE=${GSM8K_ANSWER_MODE:-"full"}
# Number of few-shot examples to prepend for GSM8K attribution (0 disables few-shot).
# Note: this only affects ATTR_DATASET=gsm8k.
NUM_FEWSHOT=${NUM_FEWSHOT:-5}

default_max_samples_for_dataset() {
  case "$1" in
    mmlu) echo "${MMLU_MAX_SAMPLES:-200}" ;;
    cmmlu) echo "${CMMLU_MAX_SAMPLES:-200}" ;;
    ceval-valid) echo "${CEVAL_VALID_MAX_SAMPLES:-200}" ;;
    gpqa_main_n_shot) echo "${GPQA_MAX_SAMPLES:-200}" ;;
    gsm8k) echo "${GSM8K_MAX_SAMPLES:-100}" ;;
    minerva_math) echo "${MINERVA_MATH_MAX_SAMPLES:-100}" ;;
    humaneval) echo "${HUMANEVAL_MAX_SAMPLES:-100}" ;;
    mbpp) echo "${MBPP_MAX_SAMPLES:-100}" ;;
    nemotron) echo "${NEMOTRON_MAX_SAMPLES:-50}" ;;
    *) echo "${DEFAULT_MAX_SAMPLES:-100}" ;;
  esac
}

run_single_dataset() {
local ATTR_DATASET="$1"
local MAX_SAMPLES="$2"
local DATA_PATH="${BASE_DATA_PATH}"

if [ "${ATTR_DATASET}" = "gpqa_main_n_shot" ] && [ -z "${DATA_PATH}" ] && [ -f "${DEFAULT_GPQA_DATA_PATH}" ]; then
  DATA_PATH="${DEFAULT_GPQA_DATA_PATH}"
fi

TAG="loss_ig_joint_${BASELINE}"
if [ "$BASELINE" = "scalar" ]; then
  TAG="loss_ig_joint_scalar${BASELINE_SCALAR}"
fi

TAG="${TAG}_maskp$(echo "${MASK_PROBS}" | tr ',' '-')_mcs${MASK_SAMPLES_PER_PROB}_${LOSS_NORMALIZE}"

OUT_DIR="${OUT_ROOT}/head_importance_${MODEL_NAME}"

# Add dataset-specific suffix (avoid confusion when sweeping)
if [ "$ATTR_DATASET" = "nemotron" ]; then
  CATEGORY_TAG=$(echo "${NEMOTRON_CATEGORIES}" | tr ',' '_')
  OUT_DIR="${OUT_DIR}_nemotron_${CATEGORY_TAG}"
elif [ "$ATTR_DATASET" = "gsm8k" ]; then
  OUT_DIR="${OUT_DIR}_gsm8k_${GSM8K_ANSWER_MODE}"
elif [ "$ATTR_DATASET" = "mmlu" ] || [ "$ATTR_DATASET" = "cmmlu" ] || [ "$ATTR_DATASET" = "ceval-valid" ] || [ "$ATTR_DATASET" = "gpqa_main_n_shot" ]; then
  SUBJECT_TAG=$(echo "${MMLU_SUBJECT}" | tr '/' '_')
  OUT_DIR="${OUT_DIR}_${ATTR_DATASET}_${SUBJECT_TAG}"
elif [ "$ATTR_DATASET" = "minerva_math" ]; then
  OUT_DIR="${OUT_DIR}_minerva_math"
elif [ "$ATTR_DATASET" = "humaneval" ]; then
  OUT_DIR="${OUT_DIR}_humaneval"
elif [ "$ATTR_DATASET" = "mbpp" ]; then
  OUT_DIR="${OUT_DIR}_mbpp"
fi

OUT_DIR="${OUT_DIR}_pm${PATH_MODE}_ts${RUN_TS}"
PROGRESS_LABEL="${ATTR_DATASET}|$(basename "${OUT_DIR}")"
mkdir -p "${OUT_DIR}"
LAST_OUT_DIR="${OUT_DIR}"

echo "Model: ${MODEL_PATH}"
echo "Out:   ${OUT_DIR}"
echo "dataset=${ATTR_DATASET} split=${SPLIT} max_samples=${MAX_SAMPLES} ig_steps=${IG_STEPS} seed=${SEED} data_seed=${DATA_SEED} mask_seed=${MASK_SEED}"
echo "data_path=${DATA_PATH:-"(hf)"}"
echo "dataset_shuffle=${DATASET_SHUFFLE}"
echo "nemotron: samples_per_category=${SAMPLES_PER_CATEGORY} pool_per_category=${NEMOTRON_POOL_PER_CATEGORY} categories=${NEMOTRON_CATEGORIES}"
echo "mmlu: subject=${MMLU_SUBJECT}"
echo "gsm8k_answer_mode=${GSM8K_ANSWER_MODE}"
echo "baseline=${BASELINE} baseline_scalar=${BASELINE_SCALAR}"
echo "mask_probs=${MASK_PROBS} mask_samples_per_prob=${MASK_SAMPLES_PER_PROB} loss_normalize=${LOSS_NORMALIZE}"
echo "ig_postprocess=${IG_POSTPROCESS} mask_batch_size=${MASK_BATCH_SIZE}"
echo "activation_checkpointing=${ACTIVATION_CHECKPOINTING}"
echo "layers=${LAYER_START}..${LAYER_END}"
echo "========================================================"

DATASET_SHUFFLE_FLAG=""
if [ "${DATASET_SHUFFLE}" = "1" ]; then
  DATASET_SHUFFLE_FLAG="--dataset_shuffle"
fi

# Set dataset_config:
# - gsm8k: config name (default main)
# - mmlu: subject (default all)
DATASET_CONFIG="main"
if [ "$ATTR_DATASET" = "mmlu" ]; then
  DATASET_CONFIG="${MMLU_SUBJECT}"
elif [ "$ATTR_DATASET" = "gpqa_main_n_shot" ]; then
  DATASET_CONFIG="gpqa_main"
elif [ "$ATTR_DATASET" = "mbpp" ]; then
  DATASET_CONFIG="sanitized"
fi

# Also capture bash-side config prints into the same run.log for reproducibility/debugging.
#
# NOTE: We intentionally redirect both stdout/stderr so tqdm warnings and errors are preserved.
{
  echo "========================================================"
  echo "[runner] Model: ${MODEL_PATH}"
  echo "[runner] Out:   ${OUT_DIR}"
  echo "[runner] dataset=${ATTR_DATASET} split=${SPLIT} max_samples=${MAX_SAMPLES} ig_steps=${IG_STEPS} seed=${SEED} data_seed=${DATA_SEED} mask_seed=${MASK_SEED}"
  echo "[runner] data_path=${DATA_PATH:-"(hf)"}"
  echo "[runner] dataset_shuffle=${DATASET_SHUFFLE}"
  echo "[runner] nemotron: samples_per_category=${SAMPLES_PER_CATEGORY} pool_per_category=${NEMOTRON_POOL_PER_CATEGORY} categories=${NEMOTRON_CATEGORIES}"
  echo "[runner] baseline=${BASELINE} baseline_scalar=${BASELINE_SCALAR}"
  echo "[runner] mask_probs=${MASK_PROBS} mask_samples_per_prob=${MASK_SAMPLES_PER_PROB} loss_normalize=${LOSS_NORMALIZE}"
  echo "[runner] ig_postprocess=${IG_POSTPROCESS} mask_batch_size=${MASK_BATCH_SIZE}"
  echo "[runner] activation_checkpointing=${ACTIVATION_CHECKPOINTING}"
  echo "[runner] progress=${SHOW_PROGRESS} progress_update_every=${PROGRESS_UPDATE_EVERY}"
  echo "[runner] progress_label=${PROGRESS_LABEL}"
  echo "[runner] layers=${LAYER_START}..${LAYER_END}"
  echo "========================================================"
  python /home/qiheng/Projects/adaptive-dllm/models/LLaDA/attribution/loss_attribution/compute_loss_attribution_all_heads.py \
    --model_path "${MODEL_PATH}" \
    --dataset "${ATTR_DATASET}" \
    --dataset_config "${DATASET_CONFIG}" \
    --data_path "${DATA_PATH}" \
    --split "${SPLIT}" \
    --max_samples "${MAX_SAMPLES}" \
    ${DATASET_SHUFFLE_FLAG} \
    --samples_per_category "${SAMPLES_PER_CATEGORY}" \
    --nemotron_pool_per_category "${NEMOTRON_POOL_PER_CATEGORY}" \
    --nemotron_categories "${NEMOTRON_CATEGORIES}" \
    --seed "${SEED}" \
    --data_seed "${DATA_SEED}" \
    --mask_seed "${MASK_SEED}" \
    --ig_steps "${IG_STEPS}" \
    --max_length "${MAX_LENGTH}" \
    --min_completion_tokens "${MIN_COMPLETION_TOKENS}" \
    --baseline "${BASELINE}" \
    --baseline_scalar "${BASELINE_SCALAR}" \
    --gsm8k_answer_mode "${GSM8K_ANSWER_MODE}" \
    --num_fewshot "${NUM_FEWSHOT}" \
    --mask_probs "${MASK_PROBS}" \
    --mask_samples_per_prob "${MASK_SAMPLES_PER_PROB}" \
    --loss_normalize "${LOSS_NORMALIZE}" \
    --ig_postprocess "${IG_POSTPROCESS}" \
    --mask_batch_size "${MASK_BATCH_SIZE}" \
    --path_mode "${PATH_MODE}" \
    --path_samples "${PATH_SAMPLES}" \
    --path_seed "${PATH_SEED}" \
    --activation_checkpointing "${ACTIVATION_CHECKPOINTING}" \
    --layer_start "${LAYER_START}" \
    --layer_end "${LAYER_END}" \
    --output_dir "${OUT_DIR}" \
    --progress_label "${PROGRESS_LABEL}" \
    --use_amp_bf16 \
    ${PROGRESS_FLAG} \
    --debug_dump_samples "${DEBUG_DUMP_SAMPLES}" \
    --debug_save_per_sample "${DEBUG_SAVE_PER_SAMPLE}"
} 2>&1 | tee "${OUT_DIR}/run.log"

echo "========================================================"
echo "Finished at: $(date)"
echo "Wrote: ${OUT_DIR}/head_importance.pt"
echo "Log:   ${OUT_DIR}/run.log"
echo "========================================================"
}

IFS=',' read -r -a ATTR_DATASETS <<< "${ATTR_DATASETS_STR}"
echo "Datasets to run: ${ATTR_DATASETS[*]}"
SUCCESS_DATASETS=()
SUCCESS_OUT_DIRS=()
FAILED_DATASETS=()
FAILED_CODES=()
for dataset in "${ATTR_DATASETS[@]}"; do
  dataset="$(echo "${dataset}" | xargs)"
  [ -z "${dataset}" ] && continue
  max_samples_for_dataset="${GLOBAL_MAX_SAMPLES:-$(default_max_samples_for_dataset "${dataset}")}"
  echo ""
  echo "########################################################"
  echo "Running attribution dataset: ${dataset} (max_samples=${max_samples_for_dataset})"
  echo "########################################################"
  if run_single_dataset "${dataset}" "${max_samples_for_dataset}"; then
    SUCCESS_DATASETS+=("${dataset}")
    SUCCESS_OUT_DIRS+=("${LAST_OUT_DIR}")
  else
    rc=$?
    FAILED_DATASETS+=("${dataset}")
    FAILED_CODES+=("${rc}")
    echo "[summary] FAILED dataset=${dataset} rc=${rc}"
  fi
done

echo ""
echo "========================================================"
echo "Batch Summary"
echo "========================================================"
echo "Succeeded: ${#SUCCESS_DATASETS[@]}"
for i in "${!SUCCESS_DATASETS[@]}"; do
  echo "  [OK] ${SUCCESS_DATASETS[$i]} -> ${SUCCESS_OUT_DIRS[$i]}"
done
echo "Failed: ${#FAILED_DATASETS[@]}"
for i in "${!FAILED_DATASETS[@]}"; do
  echo "  [FAIL] ${FAILED_DATASETS[$i]} (rc=${FAILED_CODES[$i]})"
done
echo "========================================================"

if [ "${#FAILED_DATASETS[@]}" -gt 0 ]; then
  exit 1
fi


