#!/bin/bash
#SBATCH --job-name=llada-loss-attr
#SBATCH --output=logs/loss_attr_%j.out
#SBATCH --error=logs/loss_attr_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

set -euo pipefail

mkdir -p logs

echo "========================================================"
echo "LLaDA Loss Attribution (Layer-wise IG)"
echo "========================================================"
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-"(unset)"}"
echo "========================================================"

# Activate env (adjust if your cluster differs)
source ~/miniconda3/bin/activate adaptive-dllm

export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=/home/qiheng/Projects/adaptive-dllm:$PYTHONPATH

# Optional: pin to a specific GPU id (within allocated resources)
GPU_ID=${GPU_ID:-1}
if [ -n "$GPU_ID" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  echo "Pinned GPU via CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

MODEL_PATH=${MODEL_PATH:-"GSAI-ML/LLaDA-8B-Base"}
OUT_ROOT=${OUT_ROOT:-"/home/qiheng/Projects/adaptive-dllm/configs"}
RUN_TS=${RUN_TS:-"$(date +%Y%m%d_%H%M%S)"}

# Attribution dataset (can differ from downstream eval tasks)
# - gsm8k: uses final answer after '####' (answer-only)
# - nemotron: uses Nemotron 'output' as completion (completion-only)
ATTR_DATASET=${ATTR_DATASET:-"nemotron"}     # nemotron | gsm8k
SPLIT=${SPLIT:-"test"}                      # gsm8k only
SAMPLES_PER_CATEGORY=${SAMPLES_PER_CATEGORY:-50}  # nemotron only
NEMOTRON_CATEGORIES=${NEMOTRON_CATEGORIES:-"code,math,science,chat,safety"} # nemotron only

MAX_SAMPLES=${MAX_SAMPLES:-200}
IG_STEPS=${IG_STEPS:-8}
# MAX_LENGTH controls prompt+completion truncation. 2048 is usually a safe default for 8B on 1 GPU.
# If you have enough memory and want less truncation, set 4096 (LLaDA max is typically 4096).
MAX_LENGTH=${MAX_LENGTH:-2048}
SEED=${SEED:-1234}

# Choose baseline:
# - zero   : α from 0 -> 1
# - scalar : α from BASELINE_SCALAR -> 1 (default BASELINE_SCALAR=0.3)
BASELINE=${BASELINE:-"zero"}        # zero | scalar
BASELINE_SCALAR=${BASELINE_SCALAR:-0.3}

# Multi-timestep diffusion-style masking (NEW)
MASK_PROBS=${MASK_PROBS:-"0.3,0.5,0.7,0.9"}
MASK_SAMPLES_PER_PROB=${MASK_SAMPLES_PER_PROB:-2}
LOSS_NORMALIZE=${LOSS_NORMALIZE:-"mean_masked"}  # mean_masked | sum
IG_POSTPROCESS=${IG_POSTPROCESS:-"signed"}          # abs | signed | relu
MASK_BATCH_SIZE=${MASK_BATCH_SIZE:-1}           # 0 => all variants in one batch (may OOM)
ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING:-"whole_layer"}  # none | whole_layer | one_in_two | ...

# Layer range (inclusive). -1 means last layer.
LAYER_START=${LAYER_START:-0}
LAYER_END=${LAYER_END:-31}

TAG="loss_ig_${BASELINE}"
if [ "$BASELINE" = "scalar" ]; then
  TAG="loss_ig_scalar${BASELINE_SCALAR}"
fi

OUT_DIR="${OUT_ROOT}/head_importance_llada_base_${TAG}_seed${SEED}_n${MAX_SAMPLES}_k${IG_STEPS}"
OUT_DIR="${OUT_DIR}_ts${RUN_TS}"
mkdir -p "${OUT_DIR}"

echo "Model: ${MODEL_PATH}"
echo "Out:   ${OUT_DIR}"
echo "dataset=${ATTR_DATASET} split=${SPLIT} max_samples=${MAX_SAMPLES} ig_steps=${IG_STEPS} seed=${SEED}"
echo "nemotron: samples_per_category=${SAMPLES_PER_CATEGORY} categories=${NEMOTRON_CATEGORIES}"
echo "baseline=${BASELINE} baseline_scalar=${BASELINE_SCALAR}"
echo "mask_probs=${MASK_PROBS} mask_samples_per_prob=${MASK_SAMPLES_PER_PROB} loss_normalize=${LOSS_NORMALIZE}"
echo "ig_postprocess=${IG_POSTPROCESS} mask_batch_size=${MASK_BATCH_SIZE}"
echo "activation_checkpointing=${ACTIVATION_CHECKPOINTING}"
echo "layers=${LAYER_START}..${LAYER_END}"
echo "========================================================"

python /home/qiheng/Projects/adaptive-dllm/models/LLaDA/attribution/loss_attribution/compute_loss_attribution.py \
  --model_path "${MODEL_PATH}" \
  --dataset "${ATTR_DATASET}" \
  --dataset_config main \
  --split "${SPLIT}" \
  --max_samples "${MAX_SAMPLES}" \
  --samples_per_category "${SAMPLES_PER_CATEGORY}" \
  --nemotron_categories "${NEMOTRON_CATEGORIES}" \
  --seed "${SEED}" \
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
  --use_amp_bf16

echo "========================================================"
echo "Finished at: $(date)"
echo "Wrote: ${OUT_DIR}/head_importance.pt"
echo "========================================================"


