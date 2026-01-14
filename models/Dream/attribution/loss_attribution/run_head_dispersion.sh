#!/usr/bin/env bash
set -euo pipefail

# One-command runner for Dream head dispersion (block-level attention dispersion).
# Usage:
#   bash /home/qiheng/Projects/adaptive-dllm/models/Dream/attribution/loss_attribution/run_head_dispersion.sh
#
# Optional overrides (env vars):
#   GPU_ID=0 MODEL_PATH=... OUT_ROOT=... DISP_DATASET=gsm8k MAX_SAMPLES=200 BLOCK_SIZE=128 ...

mkdir -p logs

echo "========================================================"
echo "Dream Head Dispersion (block-level)"
echo "========================================================"
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-"(unset)"}"
echo "========================================================"

# Pin to a specific GPU id (default follows existing Dream runner)
GPU_ID=${GPU_ID:-2}
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
MODEL_NAME=${MODEL_NAME:-"dream_base"}

OUT_ROOT=${OUT_ROOT:-"/home/qiheng/Projects/adaptive-dllm/configs"}
RUN_TS=${RUN_TS:-"$(date +%Y%m%d_%H%M%S)"}

# ---------------------------
# Dataset
# ---------------------------
DISP_DATASET=${DISP_DATASET:-"nemotron"}  # nemotron | gsm8k
DATASET_CONFIG=${DATASET_CONFIG:-"main"}  # gsm8k only
SPLIT=${SPLIT:-"test"}                    # gsm8k only
MAX_SAMPLES=${MAX_SAMPLES:-50}
DATASET_SHUFFLE=${DATASET_SHUFFLE:-1}    # 1 => --dataset_shuffle

# Keep attribution/eval format consistent (optional, recommended for Dream)
USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE:-1}         # 1 => --dataset_use_chat_template
GSM8K_COMPLETION_MODE=${GSM8K_COMPLETION_MODE:-"final"}  # final|full

# ---------------------------
# Core knobs
# ---------------------------
SEED=${SEED:-1234}
DATA_SEED=${DATA_SEED:-${SEED}}
MAX_LENGTH=${MAX_LENGTH:-2048}

BLOCK_SIZE=${BLOCK_SIZE:-128}
QUERY_SPAN=${QUERY_SPAN:-"completion"}    # all | completion | last_n
LAST_N=${LAST_N:-256}                     # only used when query_span=last_n
TOPK_FRAC=${TOPK_FRAC:-0.1}
AGGREGATION=${AGGREGATION:-"key_block_mass"} # key_block_mass | qk_block_mean
CAUSAL_MASK=${CAUSAL_MASK:-0}               # 1 => --causal_mask (default off; Dream here is non-causal)

# Nemotron sampling knobs (to make data_seed meaningful)
SAMPLES_PER_CATEGORY=${SAMPLES_PER_CATEGORY:-10}
NEMOTRON_POOL_PER_CATEGORY=${NEMOTRON_POOL_PER_CATEGORY:-1000}
NEMOTRON_CATEGORIES=${NEMOTRON_CATEGORIES:-"code,math,science,chat,safety"}

USE_AMP_BF16=${USE_AMP_BF16:-1}           # 1 => --use_amp_bf16
NO_PROGRESS=${NO_PROGRESS:-0}             # 1 => --no_progress
PROGRESS_UPDATE_EVERY=${PROGRESS_UPDATE_EVERY:-10}

# Layer range (inclusive). -1 means last layer.
LAYER_START=${LAYER_START:-0}
LAYER_END=${LAYER_END:--1}

DATASET_SHUFFLE_FLAG=""
if [ "${DATASET_SHUFFLE}" = "1" ]; then
  DATASET_SHUFFLE_FLAG="--dataset_shuffle"
fi

AMP_FLAG=""
if [ "${USE_AMP_BF16}" = "1" ]; then
  AMP_FLAG="--use_amp_bf16"
fi

CAUSAL_FLAG=""
if [ "${CAUSAL_MASK}" = "1" ]; then
  CAUSAL_FLAG="--causal_mask"
fi

PROGRESS_FLAG=""
if [ "${NO_PROGRESS}" = "1" ]; then
  PROGRESS_FLAG="--no_progress"
else
  PROGRESS_FLAG="--progress_update_every ${PROGRESS_UPDATE_EVERY}"
fi

CHAT_FLAG=""
if [ "${USE_CHAT_TEMPLATE}" = "1" ]; then
  CHAT_FLAG="--dataset_use_chat_template"
fi

TAG="disp_${DISP_DATASET}_bs${BLOCK_SIZE}_qs${QUERY_SPAN}_topk${TOPK_FRAC}_${AGGREGATION}"
OUT_DIR="${OUT_ROOT}/head_dispersion_${MODEL_NAME}_${TAG}_seed${SEED}_dseed${DATA_SEED}_n${MAX_SAMPLES}_L${MAX_LENGTH}_ts${RUN_TS}"
mkdir -p "${OUT_DIR}"

{
  echo "Model: ${MODEL_PATH}"
  echo "Out:   ${OUT_DIR}"
  echo "dataset=${DISP_DATASET} split=${SPLIT} max_samples=${MAX_SAMPLES} seed=${SEED} data_seed=${DATA_SEED} max_length=${MAX_LENGTH}"
  echo "dataset_shuffle=${DATASET_SHUFFLE}"
  echo "nemotron: samples_per_category=${SAMPLES_PER_CATEGORY} pool_per_category=${NEMOTRON_POOL_PER_CATEGORY} categories=${NEMOTRON_CATEGORIES}"
  echo "block_size=${BLOCK_SIZE} query_span=${QUERY_SPAN} last_n=${LAST_N} topk_frac=${TOPK_FRAC}"
  echo "aggregation=${AGGREGATION} causal_mask=${CAUSAL_MASK}"
  echo "use_chat_template=${USE_CHAT_TEMPLATE} gsm8k_completion_mode=${GSM8K_COMPLETION_MODE}"
  echo "layers=${LAYER_START}..${LAYER_END}"
  echo "========================================================"

  python /home/qiheng/Projects/adaptive-dllm/models/Dream/attribution/loss_attribution/compute_head_dispersion.py \
    --model_path "${MODEL_PATH}" \
    --dataset "${DISP_DATASET}" \
    --dataset_config "${DATASET_CONFIG}" \
    --split "${SPLIT}" \
    --max_samples "${MAX_SAMPLES}" \
    ${DATASET_SHUFFLE_FLAG} \
    --seed "${SEED}" \
    --data_seed "${DATA_SEED}" \
    --max_length "${MAX_LENGTH}" \
    --block_size "${BLOCK_SIZE}" \
    --query_span "${QUERY_SPAN}" \
    --last_n "${LAST_N}" \
    --topk_frac "${TOPK_FRAC}" \
    --aggregation "${AGGREGATION}" \
    ${CAUSAL_FLAG} \
    --samples_per_category "${SAMPLES_PER_CATEGORY}" \
    --nemotron_pool_per_category "${NEMOTRON_POOL_PER_CATEGORY}" \
    --nemotron_categories "${NEMOTRON_CATEGORIES}" \
    ${CHAT_FLAG} \
    --gsm8k_completion_mode "${GSM8K_COMPLETION_MODE}" \
    --layer_start "${LAYER_START}" \
    --layer_end "${LAYER_END}" \
    --output_dir "${OUT_DIR}" \
    --device "cuda" \
    ${AMP_FLAG} \
    ${PROGRESS_FLAG}
} 2>&1 | tee "${OUT_DIR}/run.log"

echo "========================================================"
echo "Finished at: $(date)"
echo "Wrote: ${OUT_DIR}/head_dispersion.pt"
echo "Log:   ${OUT_DIR}/run.log"
echo "========================================================"


