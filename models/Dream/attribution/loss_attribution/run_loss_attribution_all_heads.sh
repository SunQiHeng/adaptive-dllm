#!/usr/bin/env bash
set -euo pipefail

# One-command runner for loss-based head attribution (ALL-HEADS JOINT IG) for Dream.
# Default behavior:
#   - run all 8 unified tasks serially on one GPU
#   - use per-dataset default attribution sample counts
# Usage:
#   bash /home/qiheng/Projects/adaptive-dllm/models/Dream/attribution/loss_attribution/run_loss_attribution_all_heads.sh
#
# Useful overrides (env vars):
#   ATTR_DATASETS_STR="mmlu,gsm8k,humaneval"
#   MAX_SAMPLES=80
#   MMLU_MAX_SAMPLES=200 GSM8K_MAX_SAMPLES=100 HUMANEVAL_MAX_SAMPLES=100
#   GPQA_DATA_PATH=/path/to/gpqa_main.jsonl

mkdir -p logs

echo "========================================================"
echo "Dream Loss Attribution (All-heads Joint IG)"
echo "========================================================"
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-"(unset)"}"
echo "========================================================"

# Pin to a specific GPU id (default follows existing Dream runners)
GPU_ID=${GPU_ID:-5}
export CUDA_VISIBLE_DEVICES="$GPU_ID"
echo "Pinned GPU via CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Optional: activate conda env (non-fatal if not present)
if [ -f "$HOME/miniconda3/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/bin/activate" adaptive-dllm || true
fi

export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=/home/qiheng/Projects/adaptive-dllm:${PYTHONPATH:-}

# ---------------------------
# Model / output
# ---------------------------
MODEL_PATH=${MODEL_PATH:-"/data/qh_models/Dream-v0-Instruct-7B"}
MODEL_NAME=${MODEL_NAME:-"dream"}
DEFAULT_GPQA_DATA_PATH="/home/qiheng/Projects/adaptive-dllm/evaluation/local_data/gpqa/gpqa_main.jsonl"

OUT_ROOT=${OUT_ROOT:-"/home/qiheng/Projects/adaptive-dllm/configs"}
RUN_TS=${RUN_TS:-"$(date +%Y%m%d_%H%M%S)"}

# ---------------------------
# Attribution dataset
# ---------------------------
ATTR_DATASET=${ATTR_DATASET:-"mmlu"}   # gsm8k | minerva_math | nemotron | mmlu | cmmlu | ceval-valid | gpqa_main_n_shot | humaneval | mbpp
ATTR_DATASETS_STR=${ATTR_DATASETS_STR:-"mmlu,cmmlu,ceval-valid,gsm8k,minerva_math,gpqa_main_n_shot,humaneval,mbpp"}
SPLIT=${SPLIT:-"test"}                 # gsm8k/mmlu split; humaneval is fixed test internally
DATA_PATH=${DATA_PATH:-""}
BASE_DATA_PATH=${DATA_PATH:-""}

SAMPLES_PER_CATEGORY=${SAMPLES_PER_CATEGORY:-10}           # nemotron only
NEMOTRON_CATEGORIES=${NEMOTRON_CATEGORIES:-"code,math,science,chat,safety"} # nemotron only
NEMOTRON_POOL_PER_CATEGORY=${NEMOTRON_POOL_PER_CATEGORY:-1000}             # nemotron only
MMLU_SUBJECT=${MMLU_SUBJECT:-"all"}                         # mmlu only

USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE:-1}  # 1 => --use_chat_template

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

# ---------------------------
# Core knobs
# ---------------------------
GLOBAL_MAX_SAMPLES=${MAX_SAMPLES:-""}
IG_STEPS=${IG_STEPS:-8}
MAX_LENGTH=${MAX_LENGTH:-2048}

SEED=${SEED:-47}
DATA_SEED=${DATA_SEED:-${SEED}}
MASK_SEED=${MASK_SEED:-${SEED}}

# IG baseline:
BASELINE=${BASELINE:-"zero"}        # zero | scalar
BASELINE_SCALAR=${BASELINE_SCALAR:-0.3}

# Multi-timestep diffusion-style masking
MASK_PROBS=${MASK_PROBS:-"0.15,0.3,0.5,0.7,0.9"}
MASK_SAMPLES_PER_PROB=${MASK_SAMPLES_PER_PROB:-2}
LOSS_NORMALIZE=${LOSS_NORMALIZE:-"mean_masked"}  # mean_masked | sum
IG_POSTPROCESS=${IG_POSTPROCESS:-"signed"}       # abs | signed | relu
MASK_BATCH_SIZE=${MASK_BATCH_SIZE:-1}            # 0 => all variants in one batch (may OOM)

# Integrated path mode for joint IG (diagonal vs randomized path)
PATH_MODE=${PATH_MODE:-"random_threshold"}                 # diagonal | random_threshold
PATH_SAMPLES=${PATH_SAMPLES:-25}                    # >1 only meaningful for random_threshold
PATH_SEED=${PATH_SEED:--1}                         # -1 => use mask_seed

# Dream: joint attribution still benefits from gradient checkpointing
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}  # 1 => --gradient_checkpointing

# Progress / debug
SHOW_PROGRESS=${SHOW_PROGRESS:-1}               # 1 => --show_progress
PROGRESS_UPDATE_EVERY=${PROGRESS_UPDATE_EVERY:-10}
DEBUG_GATE=${DEBUG_GATE:-0}                    # 1 => --debug_gate

# Layer range (inclusive). -1 means last layer.
LAYER_START=${LAYER_START:-0}
LAYER_END=${LAYER_END:--1}

# GSM8K attribution target:
# - final:      supervise only final answer tokens (after '####')
# - final_hash: supervise "#### <final>" (closer to lm-eval extraction pattern)
# - full:       supervise full `answer` field (rationale + final), usually more stable
GSM8K_ANSWER_MODE=${GSM8K_ANSWER_MODE:-"full"}

# ---------------------------
# Flags
# ---------------------------
CHAT_FLAG=""
if [ "${USE_CHAT_TEMPLATE}" = "1" ]; then
  CHAT_FLAG="--use_chat_template"
fi

GC_FLAG=""
if [ "${GRADIENT_CHECKPOINTING}" = "1" ]; then
  GC_FLAG="--gradient_checkpointing"
fi

PROGRESS_FLAG=""
if [ "${SHOW_PROGRESS}" = "1" ]; then
  PROGRESS_FLAG="--show_progress --progress_update_every ${PROGRESS_UPDATE_EVERY}"
fi

DEBUG_FLAG=""
if [ "${DEBUG_GATE}" = "1" ]; then
  DEBUG_FLAG="--debug_gate"
fi

# dataset_config is used by compute script:
# - gsm8k: config name (default main)
# - mmlu: subject (default all)
DATASET_CONFIG="main"
if [ "${ATTR_DATASET}" = "mmlu" ]; then
  DATASET_CONFIG="${MMLU_SUBJECT}"
elif [ "${ATTR_DATASET}" = "gpqa_main_n_shot" ]; then
  DATASET_CONFIG="gpqa_main"
elif [ "${ATTR_DATASET}" = "mbpp" ]; then
  DATASET_CONFIG="sanitized"
fi

# ---------------------------
# Output dir (align with LLaDA runner naming style)
# ---------------------------
TAG="loss_ig_joint_${BASELINE}"
if [ "${BASELINE}" = "scalar" ]; then
  TAG="loss_ig_joint_scalar${BASELINE_SCALAR}"
fi
TAG="${TAG}_maskp$(echo "${MASK_PROBS}" | tr ',' '-')_mcs${MASK_SAMPLES_PER_PROB}_${LOSS_NORMALIZE}"

OUT_DIR="${OUT_ROOT}/head_importance_${MODEL_NAME}"
if [ "${ATTR_DATASET}" = "nemotron" ]; then
  CATEGORY_TAG=$(echo "${NEMOTRON_CATEGORIES}" | tr ',' '_')
  OUT_DIR="${OUT_DIR}_nemotron_${CATEGORY_TAG}"
elif [ "${ATTR_DATASET}" = "gsm8k" ]; then
  OUT_DIR="${OUT_DIR}_gsm8k_${GSM8K_ANSWER_MODE}"
elif [ "${ATTR_DATASET}" = "mmlu" ] || [ "${ATTR_DATASET}" = "cmmlu" ] || [ "${ATTR_DATASET}" = "ceval-valid" ] || [ "${ATTR_DATASET}" = "gpqa_main_n_shot" ]; then
  SUBJECT_TAG=$(echo "${MMLU_SUBJECT}" | tr '/' '_')
  OUT_DIR="${OUT_DIR}_${ATTR_DATASET}_${SUBJECT_TAG}"
elif [ "${ATTR_DATASET}" = "minerva_math" ]; then
  OUT_DIR="${OUT_DIR}_minerva_math"
elif [ "${ATTR_DATASET}" = "humaneval" ]; then
  OUT_DIR="${OUT_DIR}_humaneval"
elif [ "${ATTR_DATASET}" = "mbpp" ]; then
  OUT_DIR="${OUT_DIR}_mbpp"
fi
OUT_DIR="${OUT_DIR}_pm${PATH_MODE}_ts${RUN_TS}"
PROGRESS_LABEL="${ATTR_DATASET}|$(basename "${OUT_DIR}")"
mkdir -p "${OUT_DIR}"
LAST_OUT_DIR="${OUT_DIR}"

echo "Model: ${MODEL_PATH}"
echo "Out:   ${OUT_DIR}"
echo "dataset=${ATTR_DATASET} config=${DATASET_CONFIG} split=${SPLIT} max_samples=${MAX_SAMPLES}"
echo "data_path=${DATA_PATH:-"(hf)"}"
echo "seed=${SEED} data_seed=${DATA_SEED} mask_seed=${MASK_SEED}"
echo "ig_steps=${IG_STEPS} max_length=${MAX_LENGTH} path_mode=${PATH_MODE} path_samples=${PATH_SAMPLES} path_seed=${PATH_SEED}"
echo "use_chat_template=${USE_CHAT_TEMPLATE} gradient_checkpointing=${GRADIENT_CHECKPOINTING}"
echo "baseline=${BASELINE} baseline_scalar=${BASELINE_SCALAR}"
echo "mask_probs=${MASK_PROBS} mask_samples_per_prob=${MASK_SAMPLES_PER_PROB} loss_normalize=${LOSS_NORMALIZE}"
echo "ig_postprocess=${IG_POSTPROCESS} mask_batch_size=${MASK_BATCH_SIZE}"
echo "progress=${SHOW_PROGRESS} progress_update_every=${PROGRESS_UPDATE_EVERY} progress_label=${PROGRESS_LABEL}"
echo "layers=${LAYER_START}..${LAYER_END}"
echo "========================================================"

# Capture both stdout/stderr into run.log for reproducibility/debugging.
{
  echo "========================================================"
  echo "[runner] Model: ${MODEL_PATH}"
  echo "[runner] Out:   ${OUT_DIR}"
  echo "[runner] dataset=${ATTR_DATASET} config=${DATASET_CONFIG} split=${SPLIT} max_samples=${MAX_SAMPLES}"
  echo "[runner] data_path=${DATA_PATH:-"(hf)"}"
  echo "[runner] seed=${SEED} data_seed=${DATA_SEED} mask_seed=${MASK_SEED}"
  echo "[runner] ig_steps=${IG_STEPS} max_length=${MAX_LENGTH} path_mode=${PATH_MODE} path_samples=${PATH_SAMPLES} path_seed=${PATH_SEED}"
  echo "[runner] use_chat_template=${USE_CHAT_TEMPLATE} gradient_checkpointing=${GRADIENT_CHECKPOINTING}"
  echo "[runner] baseline=${BASELINE} baseline_scalar=${BASELINE_SCALAR}"
  echo "[runner] mask_probs=${MASK_PROBS} mask_samples_per_prob=${MASK_SAMPLES_PER_PROB} loss_normalize=${LOSS_NORMALIZE}"
  echo "[runner] ig_postprocess=${IG_POSTPROCESS} mask_batch_size=${MASK_BATCH_SIZE}"
  echo "[runner] progress=${SHOW_PROGRESS} progress_update_every=${PROGRESS_UPDATE_EVERY} progress_label=${PROGRESS_LABEL}"
  echo "[runner] layers=${LAYER_START}..${LAYER_END}"
  echo "========================================================"

  python /home/qiheng/Projects/adaptive-dllm/models/Dream/attribution/loss_attribution/compute_loss_attribution_all_heads.py \
    --model_path "${MODEL_PATH}" \
    --dataset "${ATTR_DATASET}" \
    --dataset_config "${DATASET_CONFIG}" \
    --data_path "${DATA_PATH}" \
    --split "${SPLIT}" \
    --max_samples "${MAX_SAMPLES}" \
    --samples_per_category "${SAMPLES_PER_CATEGORY}" \
    --nemotron_pool_per_category "${NEMOTRON_POOL_PER_CATEGORY}" \
    --nemotron_categories "${NEMOTRON_CATEGORIES}" \
    ${CHAT_FLAG} \
    --gsm8k_answer_mode "${GSM8K_ANSWER_MODE}" \
    --seed "${SEED}" \
    --data_seed "${DATA_SEED}" \
    --mask_seed "${MASK_SEED}" \
    --ig_steps "${IG_STEPS}" \
    --max_length "${MAX_LENGTH}" \
    --baseline "${BASELINE}" \
    --baseline_scalar "${BASELINE_SCALAR}" \
    --mask_probs "${MASK_PROBS}" \
    --mask_samples_per_prob "${MASK_SAMPLES_PER_PROB}" \
    --loss_normalize "${LOSS_NORMALIZE}" \
    --ig_postprocess "${IG_POSTPROCESS}" \
    --mask_batch_size "${MASK_BATCH_SIZE}" \
    --path_mode "${PATH_MODE}" \
    --path_samples "${PATH_SAMPLES}" \
    --path_seed "${PATH_SEED}" \
    --progress_label "${PROGRESS_LABEL}" \
    ${PROGRESS_FLAG} \
    ${DEBUG_FLAG} \
    ${GC_FLAG} \
    --layer_start "${LAYER_START}" \
    --layer_end "${LAYER_END}" \
    --output_dir "${OUT_DIR}" \
    --use_amp_bf16
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

