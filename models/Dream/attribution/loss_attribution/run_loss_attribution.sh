#!/usr/bin/env bash
set -euo pipefail

# One-command runner for Dream loss-based head attribution (layer-wise IG).
# Usage:
#   nohup bash /home/qiheng/Projects/adaptive-dllm/models/Dream/attribution/loss_attribution/run_loss_attribution.sh > logs/nohup_dream_loss_attr.log 2>&1 &
#
# Optional overrides (env vars):
#   GPU_ID=0 MODEL_PATH=... OUT_ROOT=... ATTR_DATASET=nemotron MAX_SAMPLES=200 ...

mkdir -p logs

echo "========================================================"
echo "Dream Loss Attribution (Layer-wise IG)"
echo "========================================================"
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-"(unset)"}"
echo "========================================================"

# Pin to a specific GPU id (default: 0)
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
OUT_ROOT=${OUT_ROOT:-"/home/qiheng/Projects/adaptive-dllm/configs"}
RUN_TS=${RUN_TS:-"$(date +%Y%m%d_%H%M%S)"}

# ---------------------------
# Attribution dataset
# ---------------------------
ATTR_DATASET=${ATTR_DATASET:-"nemotron"}     # nemotron | gsm8k
SPLIT=${SPLIT:-"test"}                      # gsm8k only
SAMPLES_PER_CATEGORY=${SAMPLES_PER_CATEGORY:-10}  # nemotron only
NEMOTRON_CATEGORIES=${NEMOTRON_CATEGORIES:-"code,math,science,chat,safety"} # nemotron only
NEMOTRON_POOL_PER_CATEGORY=${NEMOTRON_POOL_PER_CATEGORY:-1000}
USE_CHAT_TEMPLATE=${USE_CHAT_TEMPLATE:-1}   # 1 => --use_chat_template

# ---------------------------
# Core knobs
# ---------------------------
MAX_SAMPLES=${MAX_SAMPLES:-50}
IG_STEPS=${IG_STEPS:-8}
MAX_LENGTH=${MAX_LENGTH:-2048}
SEED=${SEED:-47}
DATA_SEED=${DATA_SEED:-${SEED}}
MASK_SEED=${MASK_SEED:-${SEED}}

BASELINE=${BASELINE:-"zero"}        # zero | scalar
BASELINE_SCALAR=${BASELINE_SCALAR:-0.3}

MASK_PROBS=${MASK_PROBS:-"0.15,0.3,0.5,0.7,0.9"}
MASK_SAMPLES_PER_PROB=${MASK_SAMPLES_PER_PROB:-2}
LOSS_NORMALIZE=${LOSS_NORMALIZE:-"mean_masked"}  # mean_masked | sum
IG_POSTPROCESS=${IG_POSTPROCESS:-"signed"}        # abs | signed | relu
MASK_BATCH_SIZE=${MASK_BATCH_SIZE:-1}             # 0 => all variants in one batch (may OOM)

# Dream 在长序列(如 2048)下做反传时激活非常吃显存；开启 gradient checkpointing 通常是必要的。
# 1 => --gradient_checkpointing (Dream 实现要求 train() 才会生效；本脚本内会处理)
# IMPORTANT: Force enable for OOM prevention
GRADIENT_CHECKPOINTING=1

# Progress (useful for nohup logs)
SHOW_PROGRESS=${SHOW_PROGRESS:-1}                 # 1 => --show_progress
PROGRESS_UPDATE_EVERY=${PROGRESS_UPDATE_EVERY:-5} # fallback print frequency when tqdm isn't used

# Layer range (inclusive). -1 means last layer.
# NOTE: Dream-v0-Instruct-7B in this repo typically has 28 layers, so default to -1 to avoid out-of-range.
LAYER_START=${LAYER_START:-0}
LAYER_END=${LAYER_END:--1}

TAG="loss_gateIG_${BASELINE}"
if [ "$BASELINE" = "scalar" ]; then
  TAG="loss_gateIG_scalar${BASELINE_SCALAR}"
fi
TAG="${TAG}_maskp$(echo "${MASK_PROBS}" | tr ',' '-')_mcs${MASK_SAMPLES_PER_PROB}_${LOSS_NORMALIZE}"

OUT_DIR="${OUT_ROOT}/head_importance_dream_base_${TAG}_seed${SEED}_n${MAX_SAMPLES}_k${IG_STEPS}_L${MAX_LENGTH}"
OUT_DIR="${OUT_DIR}_dseed${DATA_SEED}_mseed${MASK_SEED}"
OUT_DIR="${OUT_DIR}_ts${RUN_TS}"
mkdir -p "${OUT_DIR}"

echo "Model: ${MODEL_PATH}"
echo "Out:   ${OUT_DIR}"
echo "dataset=${ATTR_DATASET} split=${SPLIT} max_samples=${MAX_SAMPLES} ig_steps=${IG_STEPS} seed=${SEED} data_seed=${DATA_SEED} mask_seed=${MASK_SEED}"
echo "nemotron: samples_per_category=${SAMPLES_PER_CATEGORY} pool_per_category=${NEMOTRON_POOL_PER_CATEGORY} categories=${NEMOTRON_CATEGORIES}"
echo "baseline=${BASELINE} baseline_scalar=${BASELINE_SCALAR}"
echo "mask_probs=${MASK_PROBS} mask_samples_per_prob=${MASK_SAMPLES_PER_PROB} loss_normalize=${LOSS_NORMALIZE}"
echo "ig_postprocess=${IG_POSTPROCESS} mask_batch_size=${MASK_BATCH_SIZE}"
echo "gradient_checkpointing=${GRADIENT_CHECKPOINTING}"
echo "show_progress=${SHOW_PROGRESS} progress_update_every=${PROGRESS_UPDATE_EVERY}"
echo "layers=${LAYER_START}..${LAYER_END}"
echo "========================================================"

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

python /home/qiheng/Projects/adaptive-dllm/models/Dream/attribution/loss_attribution/compute_loss_attribution.py \
  --model_path "${MODEL_PATH}" \
  --dataset "${ATTR_DATASET}" \
  --split "${SPLIT}" \
  --max_samples "${MAX_SAMPLES}" \
  --samples_per_category "${SAMPLES_PER_CATEGORY}" \
  --nemotron_pool_per_category "${NEMOTRON_POOL_PER_CATEGORY}" \
  --nemotron_categories "${NEMOTRON_CATEGORIES}" \
  ${CHAT_FLAG} \
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
  ${PROGRESS_FLAG} \
  ${GC_FLAG} \
  --layer_start "${LAYER_START}" \
  --layer_end "${LAYER_END}" \
  --output_dir "${OUT_DIR}" \
  --use_amp_bf16 | tee "${OUT_DIR}/run.log"

echo "========================================================"
echo "Finished at: $(date)"
echo "Wrote: ${OUT_DIR}/head_importance.pt"
echo "Log:   ${OUT_DIR}/run.log"
echo "========================================================"


