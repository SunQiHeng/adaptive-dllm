#!/usr/bin/env bash
set -euo pipefail

# One-command runner for loss-based head attribution (ALL-HEADS JOINT IG) for Dream.
#
# Usage:
#   bash /home/qiheng/Projects/adaptive-dllm/models/Dream/attribution/loss_attribution/run_loss_attribution_all_heads.sh
#
# Optional overrides (env vars):
#   GPU_ID=5 MODEL_PATH=... OUT_ROOT=... ATTR_DATASET=gsm8k MAX_SAMPLES=50 IG_STEPS=8 MAX_LENGTH=2048 ...
#
# Examples:
#   ATTR_DATASET=gsm8k GSM8K_ANSWER_MODE=full PATH_MODE=random_threshold PATH_SAMPLES=4 bash run_loss_attribution_all_heads.sh
#   ATTR_DATASET=mmlu MMLU_SUBJECT=all MAX_SAMPLES=200 bash run_loss_attribution_all_heads.sh

mkdir -p logs

echo "========================================================"
echo "Dream Loss Attribution (All-heads Joint IG)"
echo "========================================================"
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-"(unset)"}"
echo "========================================================"

# Pin to a specific GPU id (default follows existing Dream runners)
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

# ---------------------------
# Model / output
# ---------------------------
MODEL_PATH=${MODEL_PATH:-"/data/qh_models/Dream-v0-Instruct-7B"}
MODEL_NAME=${MODEL_NAME:-"dream"}

OUT_ROOT=${OUT_ROOT:-"/home/qiheng/Projects/adaptive-dllm/configs"}
RUN_TS=${RUN_TS:-"$(date +%Y%m%d_%H%M%S)"}

# ---------------------------
# Attribution dataset
# ---------------------------
ATTR_DATASET=${ATTR_DATASET:-"humaneval"}   # gsm8k | nemotron | mmlu | humaneval
SPLIT=${SPLIT:-"test"}                 # gsm8k/mmlu split; humaneval is fixed test internally

SAMPLES_PER_CATEGORY=${SAMPLES_PER_CATEGORY:-10}           # nemotron only
NEMOTRON_CATEGORIES=${NEMOTRON_CATEGORIES:-"code,math,science,chat,safety"} # nemotron only
NEMOTRON_POOL_PER_CATEGORY=${NEMOTRON_POOL_PER_CATEGORY:-1000}             # nemotron only
MMLU_SUBJECT=${MMLU_SUBJECT:-"all"}                         # mmlu only

USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE:-1}  # 1 => --use_chat_template

# ---------------------------
# Core knobs
# ---------------------------
MAX_SAMPLES=${MAX_SAMPLES:-50}
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
PATH_SAMPLES=${PATH_SAMPLES:-1}                    # >1 only meaningful for random_threshold
PATH_SEED=${PATH_SEED:--1}                         # -1 => use mask_seed

# Dream: joint attribution still benefits from gradient checkpointing
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}  # 1 => --gradient_checkpointing

# Progress / debug
SHOW_PROGRESS=${SHOW_PROGRESS:-0}               # 1 => --show_progress
PROGRESS_UPDATE_EVERY=${PROGRESS_UPDATE_EVERY:-20}
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
elif [ "${ATTR_DATASET}" = "mmlu" ]; then
  SUBJECT_TAG=$(echo "${MMLU_SUBJECT}" | tr '/' '_')
  OUT_DIR="${OUT_DIR}_mmlu_${SUBJECT_TAG}"
elif [ "${ATTR_DATASET}" = "humaneval" ]; then
  OUT_DIR="${OUT_DIR}_humaneval"
fi
OUT_DIR="${OUT_DIR}_pm${PATH_MODE}_ts${RUN_TS}"
mkdir -p "${OUT_DIR}"

echo "Model: ${MODEL_PATH}"
echo "Out:   ${OUT_DIR}"
echo "dataset=${ATTR_DATASET} config=${DATASET_CONFIG} split=${SPLIT} max_samples=${MAX_SAMPLES}"
echo "seed=${SEED} data_seed=${DATA_SEED} mask_seed=${MASK_SEED}"
echo "ig_steps=${IG_STEPS} max_length=${MAX_LENGTH} path_mode=${PATH_MODE} path_samples=${PATH_SAMPLES} path_seed=${PATH_SEED}"
echo "use_chat_template=${USE_CHAT_TEMPLATE} gradient_checkpointing=${GRADIENT_CHECKPOINTING}"
echo "baseline=${BASELINE} baseline_scalar=${BASELINE_SCALAR}"
echo "mask_probs=${MASK_PROBS} mask_samples_per_prob=${MASK_SAMPLES_PER_PROB} loss_normalize=${LOSS_NORMALIZE}"
echo "ig_postprocess=${IG_POSTPROCESS} mask_batch_size=${MASK_BATCH_SIZE}"
echo "layers=${LAYER_START}..${LAYER_END}"
echo "========================================================"

# Capture both stdout/stderr into run.log for reproducibility/debugging.
{
  echo "========================================================"
  echo "[runner] Model: ${MODEL_PATH}"
  echo "[runner] Out:   ${OUT_DIR}"
  echo "[runner] dataset=${ATTR_DATASET} config=${DATASET_CONFIG} split=${SPLIT} max_samples=${MAX_SAMPLES}"
  echo "[runner] seed=${SEED} data_seed=${DATA_SEED} mask_seed=${MASK_SEED}"
  echo "[runner] ig_steps=${IG_STEPS} max_length=${MAX_LENGTH} path_mode=${PATH_MODE} path_samples=${PATH_SAMPLES} path_seed=${PATH_SEED}"
  echo "[runner] use_chat_template=${USE_CHAT_TEMPLATE} gradient_checkpointing=${GRADIENT_CHECKPOINTING}"
  echo "[runner] baseline=${BASELINE} baseline_scalar=${BASELINE_SCALAR}"
  echo "[runner] mask_probs=${MASK_PROBS} mask_samples_per_prob=${MASK_SAMPLES_PER_PROB} loss_normalize=${LOSS_NORMALIZE}"
  echo "[runner] ig_postprocess=${IG_POSTPROCESS} mask_batch_size=${MASK_BATCH_SIZE}"
  echo "[runner] layers=${LAYER_START}..${LAYER_END}"
  echo "========================================================"

  python /home/qiheng/Projects/adaptive-dllm/models/Dream/attribution/loss_attribution/compute_loss_attribution_all_heads.py \
    --model_path "${MODEL_PATH}" \
    --dataset "${ATTR_DATASET}" \
    --dataset_config "${DATASET_CONFIG}" \
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

