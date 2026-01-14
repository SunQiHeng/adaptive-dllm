#!/usr/bin/env bash
set -euo pipefail

# One-command runner for loss-based head attribution (layer-wise IG).
# Usage:
#   bash /home/qiheng/Projects/adaptive-dllm/models/LLaDA/attribution/loss_attribution/run_loss_attribution.sh
#
# Optional overrides (env vars):
#   GPU_ID=0 MODEL_PATH=... OUT_ROOT=... ATTR_DATASET=nemotron MAX_SAMPLES=200 ...

mkdir -p logs

echo "========================================================"
echo "LLaDA Loss Attribution (Layer-wise IG)"
echo "========================================================"
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-"(unset)"}"
echo "========================================================"

# Pin to a specific GPU id (default: 1)
# If you want a different GPU later, you can still override via env: GPU_ID=0 bash run_loss_attribution.sh
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

MODEL_PATH=${MODEL_PATH:-"/data/qh_models/LLaDA-1.5"}
MODEL_NAME=${MODEL_NAME:-"llada-1_5"}

# Where to write results
OUT_ROOT=${OUT_ROOT:-"/home/qiheng/Projects/adaptive-dllm/configs"}
RUN_TS=${RUN_TS:-"$(date +%Y%m%d_%H%M%S)"}

# Attribution dataset (can differ from downstream eval tasks)
ATTR_DATASET=${ATTR_DATASET:-"nemotron"}     # nemotron | gsm8k
SPLIT=${SPLIT:-"test"}                      # gsm8k only
SAMPLES_PER_CATEGORY=${SAMPLES_PER_CATEGORY:-10}  # nemotron only
NEMOTRON_CATEGORIES=${NEMOTRON_CATEGORIES:-"code,math,science,chat,safety"} # nemotron only

MAX_SAMPLES=${MAX_SAMPLES:-50}
IG_STEPS=${IG_STEPS:-8}
MAX_LENGTH=${MAX_LENGTH:-2048}
SEED=${SEED:-1234}

# Seeds:
# - DATA_SEED controls which samples are selected (dataset subsampling/shuffle).
# - MASK_SEED controls random masking positions (diffusion-style masking).
# Default: follow SEED unless explicitly overridden.
DATA_SEED=${DATA_SEED:-${SEED}}
MASK_SEED=${MASK_SEED:-${SEED}}

# Whether to shuffle dataset before taking max_samples (mainly for gsm8k; for nemotron we already shuffle within the pool)
DATASET_SHUFFLE=${DATASET_SHUFFLE:-1}  # 1 => enable --dataset_shuffle, 0 => disable

# Nemotron streaming: read a larger pool per category then shuffle+take samples_per_category.
# Set this > SAMPLES_PER_CATEGORY if you want different DATA_SEED to produce different subsets.
NEMOTRON_POOL_PER_CATEGORY=${NEMOTRON_POOL_PER_CATEGORY:-1000}

# IG baseline:
BASELINE=${BASELINE:-"zero"}        # zero | scalar
BASELINE_SCALAR=${BASELINE_SCALAR:-0.3}

# Multi-timestep diffusion-style masking (NEW)
# - mask_probs: different mask strengths, averaged (simulates multiple diffusion steps)
# - samples_per_prob: Monte-Carlo masks per strength
# - loss_normalize: recommended "mean_masked" to compare across strengths
MASK_PROBS=${MASK_PROBS:-"0.15,0.3,0.5,0.7,0.9"}
MASK_SAMPLES_PER_PROB=${MASK_SAMPLES_PER_PROB:-2}
LOSS_NORMALIZE=${LOSS_NORMALIZE:-"mean_masked"}  # mean_masked | sum
IG_POSTPROCESS=${IG_POSTPROCESS:-"signed"}          # abs | signed | relu
MASK_BATCH_SIZE=${MASK_BATCH_SIZE:-2}           # 0 => all variants in one batch (may OOM)
ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING:-"whole_layer"}  # none | whole_layer | one_in_two | ...

# Layer range (inclusive). -1 means last layer.
LAYER_START=${LAYER_START:-0}
LAYER_END=${LAYER_END:-31}

TAG="loss_ig_${BASELINE}"
if [ "$BASELINE" = "scalar" ]; then
  TAG="loss_ig_scalar${BASELINE_SCALAR}"
fi

TAG="${TAG}_maskp$(echo "${MASK_PROBS}" | tr ',' '-')_mcs${MASK_SAMPLES_PER_PROB}_${LOSS_NORMALIZE}"

OUT_DIR="${OUT_ROOT}/head_importance_${MODEL_NAME}_${TAG}"
OUT_DIR="${OUT_DIR}_ts${RUN_TS}"
mkdir -p "${OUT_DIR}"

echo "Model: ${MODEL_PATH}"
echo "Out:   ${OUT_DIR}"
echo "dataset=${ATTR_DATASET} split=${SPLIT} max_samples=${MAX_SAMPLES} ig_steps=${IG_STEPS} seed=${SEED} data_seed=${DATA_SEED} mask_seed=${MASK_SEED}"
echo "dataset_shuffle=${DATASET_SHUFFLE}"
echo "nemotron: samples_per_category=${SAMPLES_PER_CATEGORY} pool_per_category=${NEMOTRON_POOL_PER_CATEGORY} categories=${NEMOTRON_CATEGORIES}"
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

python /home/qiheng/Projects/adaptive-dllm/models/LLaDA/attribution/loss_attribution/compute_loss_attribution.py \
  --model_path "${MODEL_PATH}" \
  --dataset "${ATTR_DATASET}" \
  --dataset_config main \
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
  --baseline "${BASELINE}" \
  --baseline_scalar "${BASELINE_SCALAR}" \
  --mask_probs "${MASK_PROBS}" \
  --mask_samples_per_prob "${MASK_SAMPLES_PER_PROB}" \
  --loss_normalize "${LOSS_NORMALIZE}" \
  --ig_postprocess "${IG_POSTPROCESS}" \
  --mask_batch_size "${MASK_BATCH_SIZE}" \
  --activation_checkpointing "${ACTIVATION_CHECKPOINTING}" \
  --layer_start "${LAYER_START}" \
  --layer_end "${LAYER_END}" \
  --output_dir "${OUT_DIR}" \
  --use_amp_bf16 | tee "${OUT_DIR}/run.log"

echo "========================================================"
echo "Finished at: $(date)"
echo "Wrote: ${OUT_DIR}/head_importance.pt"
echo "Log:   ${OUT_DIR}/run.log"
echo "========================================================"


