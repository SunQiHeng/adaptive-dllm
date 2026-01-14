#!/usr/bin/env bash
set -euo pipefail

# One-command runner for loss-based head attribution (ALL-HEADS JOINT IG).
# Usage:
#   bash /home/qiheng/Projects/adaptive-dllm/models/LLaDA/attribution/loss_attribution/run_loss_attribution_all_heads.sh
#
# Optional overrides (env vars):
#   GPU_ID=0 MODEL_PATH=... OUT_ROOT=... ATTR_DATASET=nemotron MAX_SAMPLES=200 ...

mkdir -p logs

echo "========================================================"
echo "LLaDA Loss Attribution (All-heads Joint IG)"
echo "========================================================"
echo "Started at: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-"(unset)"}"
echo "========================================================"

# Pin to a specific GPU id (default: 4; keep consistent with your existing runner)
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

MODEL_PATH=${MODEL_PATH:-"/data/qh_models/LLaDA-1.5"}
MODEL_NAME=${MODEL_NAME:-"llada-1_5"}

# Where to write results
OUT_ROOT=${OUT_ROOT:-"/home/qiheng/Projects/adaptive-dllm/configs"}
RUN_TS=${RUN_TS:-"$(date +%Y%m%d_%H%M%S)"}

# Attribution dataset (can differ from downstream eval tasks)
ATTR_DATASET=${ATTR_DATASET:-"gsm8k"}     # nemotron | gsm8k | mmlu | humaneval
SPLIT=${SPLIT:-"test"}                      # dataset split name (gsm8k/mmlu); humaneval uses fixed test
SAMPLES_PER_CATEGORY=${SAMPLES_PER_CATEGORY:-10}  # nemotron only
NEMOTRON_CATEGORIES=${NEMOTRON_CATEGORIES:-"code,math,science,chat,safety"} # nemotron only
MMLU_SUBJECT=${MMLU_SUBJECT:-"all"}        # mmlu only (e.g. abstract_algebra, anatomy, ... or 'all')

MAX_SAMPLES=${MAX_SAMPLES:-50}
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

TAG="loss_ig_joint_${BASELINE}"
if [ "$BASELINE" = "scalar" ]; then
  TAG="loss_ig_joint_scalar${BASELINE_SCALAR}"
fi

TAG="${TAG}_maskp$(echo "${MASK_PROBS}" | tr ',' '-')_mcs${MASK_SAMPLES_PER_PROB}_${LOSS_NORMALIZE}"

OUT_DIR="${OUT_ROOT}/head_importance_${MODEL_NAME}_${TAG}"

# Add dataset-specific suffix (avoid confusion when sweeping)
if [ "$ATTR_DATASET" = "nemotron" ]; then
  CATEGORY_TAG=$(echo "${NEMOTRON_CATEGORIES}" | tr ',' '_')
  OUT_DIR="${OUT_DIR}_nemotron_${CATEGORY_TAG}"
elif [ "$ATTR_DATASET" = "gsm8k" ]; then
  OUT_DIR="${OUT_DIR}_gsm8k_${GSM8K_ANSWER_MODE}"
elif [ "$ATTR_DATASET" = "mmlu" ]; then
  SUBJECT_TAG=$(echo "${MMLU_SUBJECT}" | tr '/' '_')
  OUT_DIR="${OUT_DIR}_mmlu_${SUBJECT_TAG}"
elif [ "$ATTR_DATASET" = "humaneval" ]; then
  OUT_DIR="${OUT_DIR}_humaneval"
fi

OUT_DIR="${OUT_DIR}_ts${RUN_TS}"
mkdir -p "${OUT_DIR}"

echo "Model: ${MODEL_PATH}"
echo "Out:   ${OUT_DIR}"
echo "dataset=${ATTR_DATASET} split=${SPLIT} max_samples=${MAX_SAMPLES} ig_steps=${IG_STEPS} seed=${SEED} data_seed=${DATA_SEED} mask_seed=${MASK_SEED}"
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
fi

# Also capture bash-side config prints into the same run.log for reproducibility/debugging.
#
# NOTE: We intentionally redirect both stdout/stderr so tqdm warnings and errors are preserved.
{
  echo "========================================================"
  echo "[runner] Model: ${MODEL_PATH}"
  echo "[runner] Out:   ${OUT_DIR}"
  echo "[runner] dataset=${ATTR_DATASET} split=${SPLIT} max_samples=${MAX_SAMPLES} ig_steps=${IG_STEPS} seed=${SEED} data_seed=${DATA_SEED} mask_seed=${MASK_SEED}"
  echo "[runner] dataset_shuffle=${DATASET_SHUFFLE}"
  echo "[runner] nemotron: samples_per_category=${SAMPLES_PER_CATEGORY} pool_per_category=${NEMOTRON_POOL_PER_CATEGORY} categories=${NEMOTRON_CATEGORIES}"
  echo "[runner] baseline=${BASELINE} baseline_scalar=${BASELINE_SCALAR}"
  echo "[runner] mask_probs=${MASK_PROBS} mask_samples_per_prob=${MASK_SAMPLES_PER_PROB} loss_normalize=${LOSS_NORMALIZE}"
  echo "[runner] ig_postprocess=${IG_POSTPROCESS} mask_batch_size=${MASK_BATCH_SIZE}"
  echo "[runner] activation_checkpointing=${ACTIVATION_CHECKPOINTING}"
  echo "[runner] layers=${LAYER_START}..${LAYER_END}"
  echo "========================================================"
  python /home/qiheng/Projects/adaptive-dllm/models/LLaDA/attribution/loss_attribution/compute_loss_attribution_all_heads.py \
    --model_path "${MODEL_PATH}" \
    --dataset "${ATTR_DATASET}" \
    --dataset_config "${DATASET_CONFIG}" \
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
    --use_amp_bf16 \
    --debug_dump_samples "${DEBUG_DUMP_SAMPLES}" \
    --debug_save_per_sample "${DEBUG_SAVE_PER_SAMPLE}"
} 2>&1 | tee "${OUT_DIR}/run.log"

echo "========================================================"
echo "Finished at: $(date)"
echo "Wrote: ${OUT_DIR}/head_importance.pt"
echo "Log:   ${OUT_DIR}/run.log"
echo "========================================================"


