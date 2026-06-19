#!/usr/bin/env bash
set -uo pipefail

# Matched path-design ablation for HeadIG/PolyHeadIG.
# This runner computes three attribution variants on one attribution task:
#   dp      = diagonal path
#   stp     = one random-threshold path
#   poly    = multi-path random-threshold estimator
# It then evaluates prune-most/prune-least with the explicit importance path.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_FAMILY=${MODEL_FAMILY:?Set MODEL_FAMILY=llada or dream.}
GPU_ID=${GPU_ID:?Set GPU_ID to an available physical GPU index.}
RUN_TAG=${RUN_TAG:-"path_ablation_${MODEL_FAMILY}_mmlu_$(date +%Y%m%d_%H%M%S)"}

ATTR_DATASET=${ATTR_DATASET:-"mmlu"}
TASK=${TASK:-"mmlu"}
ATTR_MAX_SAMPLES=${ATTR_MAX_SAMPLES:-40}
EVAL_LIMIT=${EVAL_LIMIT:-40}
IG_STEPS=${IG_STEPS:-8}
MASK_PROBS=${MASK_PROBS:-"0.15,0.3,0.5,0.7,0.9"}
MASK_SAMPLES_PER_PROB=${MASK_SAMPLES_PER_PROB:-2}
SEED=${SEED:-123}
DATA_SEED=${DATA_SEED:-${SEED}}
MASK_SEED=${MASK_SEED:-${SEED}}
PATH_SEED=${PATH_SEED:-${SEED}}
RUN_ADAPTIVE=${RUN_ADAPTIVE:-0}

STATE_DIR="${PROJECT_ROOT}/logs/path_ablation/${RUN_TAG}"
STATUS_FILE="${STATE_DIR}/status.tsv"
mkdir -p "${STATE_DIR}"
touch "${STATUS_FILE}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

record_status() {
  local item="$1"
  local status="$2"
  local detail="${3:-}"
  printf "%s\t%s\t%s\t%s\n" "$(timestamp)" "${item}" "${status}" "${detail}" >> "${STATUS_FILE}"
  echo "[$(timestamp)] item=${item} status=${status} ${detail}"
}

safe_name() {
  printf "%s" "$1" | tr '/.' '__' | tr '-' '_'
}

model_context() {
  case "${MODEL_FAMILY}" in
    llada)
      MODEL_NAME="llada-1_5"
      ATTR_RUNNER="${PROJECT_ROOT}/models/LLaDA/attribution/loss_attribution/run_loss_attribution_all_heads.sh"
      MASK_RUNNER="${PROJECT_ROOT}/evaluation/llada/run_eval_mask_head_task.sh"
      ADAPTIVE_RUNNER="${PROJECT_ROOT}/evaluation/llada/run_eval_task.sh"
      LAYER_START_DEFAULT=0
      LAYER_END_DEFAULT=31
      PRUNE_FRAC="0.2"
      ;;
    dream)
      MODEL_NAME="dream"
      ATTR_RUNNER="${PROJECT_ROOT}/models/Dream/attribution/loss_attribution/run_loss_attribution_all_heads.sh"
      MASK_RUNNER="${PROJECT_ROOT}/evaluation/dream/run_eval_mask_head_task.sh"
      ADAPTIVE_RUNNER="${PROJECT_ROOT}/evaluation/dream/run_eval_task.sh"
      LAYER_START_DEFAULT=0
      LAYER_END_DEFAULT=27
      PRUNE_FRAC="0.05"
      ;;
    *)
      echo "ERROR: unsupported MODEL_FAMILY=${MODEL_FAMILY}" >&2
      return 2
      ;;
  esac
}

attr_label_for_dataset() {
  case "$1" in
    mmlu) echo "mmlu_all" ;;
    cmmlu) echo "cmmlu_all" ;;
    ceval-valid) echo "ceval-valid_all" ;;
    gpqa_main_n_shot) echo "gpqa_main_n_shot_all" ;;
    gsm8k) echo "gsm8k_final_hash" ;;
    minerva_math) echo "minerva_math" ;;
    humaneval) echo "humaneval" ;;
    mbpp) echo "mbpp" ;;
    *)
      echo "ERROR: unsupported attribution dataset=$1" >&2
      return 2
      ;;
  esac
}

eval_attr_label_for_dataset() {
  case "$1" in
    gsm8k) echo "gsm8k" ;;
    *) attr_label_for_dataset "$1" ;;
  esac
}

importance_path_for_variant() {
  local path_mode="$1"
  local run_ts="$2"
  local attr_label
  attr_label="$(attr_label_for_dataset "${ATTR_DATASET}")" || return $?
  echo "${PROJECT_ROOT}/configs/aconfigs/head_importance_${MODEL_NAME}_${attr_label}_pm${path_mode}_ts${run_ts}/head_importance.pt"
}

run_stage() {
  local item="$1"
  local log_path="$2"
  shift 2
  local marker="${STATE_DIR}/${item}.done"
  if [ -f "${marker}" ]; then
    record_status "${item}" "SKIP" "already completed"
    return 0
  fi

  record_status "${item}" "RUNNING" "log=${log_path}"
  "$@" > "${log_path}" 2>&1
  local rc=$?
  if [ "${rc}" -eq 0 ]; then
    touch "${marker}"
    record_status "${item}" "DONE"
  else
    record_status "${item}" "FAILED" "rc=${rc}"
  fi
  return "${rc}"
}

run_attribution_variant() {
  local path_name="$1"
  local path_mode="$2"
  local path_samples="$3"
  local run_ts="${RUN_TAG}_${path_name}"
  local item="attr_${MODEL_FAMILY}_${path_name}_$(safe_name "${ATTR_DATASET}")"
  local log_path="${STATE_DIR}/${item}.log"

  run_stage "${item}" "${log_path}" \
    env \
      GPU_ID="${GPU_ID}" \
      ATTR_DATASETS_STR="${ATTR_DATASET}" \
      MAX_SAMPLES="${ATTR_MAX_SAMPLES}" \
      IG_STEPS="${IG_STEPS}" \
      MASK_PROBS="${MASK_PROBS}" \
      MASK_SAMPLES_PER_PROB="${MASK_SAMPLES_PER_PROB}" \
      IG_POSTPROCESS="signed" \
      LOSS_NORMALIZE="mean_masked" \
      SEED="${SEED}" \
      DATA_SEED="${DATA_SEED}" \
      MASK_SEED="${MASK_SEED}" \
      PATH_MODE="${path_mode}" \
      PATH_SAMPLES="${path_samples}" \
      PATH_SEED="${PATH_SEED}" \
      GSM8K_ANSWER_MODE="${GSM8K_ANSWER_MODE:-final_hash}" \
      RUN_TS="${run_ts}" \
      DEBUG_DUMP_SAMPLES=0 \
      DEBUG_SAVE_PER_SAMPLE=0 \
      SHOW_PROGRESS=1 \
      bash "${ATTR_RUNNER}"
}

run_pruning_variant() {
  local path_name="$1"
  local path_mode="$2"
  local run_ts="${RUN_TAG}_${path_name}"
  local importance_path
  importance_path="$(importance_path_for_variant "${path_mode}" "${run_ts}")" || return $?
  local eval_attr_label
  eval_attr_label="$(eval_attr_label_for_dataset "${ATTR_DATASET}")" || return $?
  local item="mask_${MODEL_FAMILY}_${path_name}_$(safe_name "${TASK}")"
  local log_path="${STATE_DIR}/${item}.log"
  local importance_tag="pathab_${RUN_TAG}_${MODEL_FAMILY}_${path_name}_$(safe_name "${TASK}")"
  local mc_num=1
  if [ "${TASK}" = "gpqa_main_n_shot" ]; then
    mc_num="${GPQA_MC_NUM:-8}"
  fi

  if [ ! -f "${importance_path}" ]; then
    record_status "${item}" "FAILED" "importance not found: ${importance_path}"
    return 3
  fi

  if [ "${MODEL_FAMILY}" = "llada" ]; then
    run_stage "${item}" "${log_path}" \
      env \
        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        HF_HUB_OFFLINE=1 \
        HF_DATASETS_OFFLINE=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        MODEL_NAME="${MODEL_NAME}" \
        ATTR_METHOD=headig \
        ATTR_DATASETS_STR="${eval_attr_label}" \
        TASKS_STR="${TASK}" \
        LIMIT="${EVAL_LIMIT}" \
        IMPORTANCE_PATH="${importance_path}" \
        IMPORTANCE_TAG="${importance_tag}" \
        USE_NEGATED_MODES_STR=1 \
        MODES="most,least" \
        PRUNE_SCOPE=layer \
        PRUNE_K_FRAC="${PRUNE_FRAC}" \
        LAYER_START="${LAYER_START_DEFAULT}" \
        LAYER_END="${LAYER_END_DEFAULT}" \
        MC_NUM="${mc_num}" \
        bash "${MASK_RUNNER}"
  else
    run_stage "${item}" "${log_path}" \
      env \
        CUDA_VISIBLE_DEVICES="${GPU_ID}" \
        HF_HUB_OFFLINE=1 \
        HF_DATASETS_OFFLINE=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        MODEL_NAME="${MODEL_NAME}" \
        ATTR_METHOD=headig \
        ATTR_DATASETS_STR="${eval_attr_label}" \
        TASKS_STR="${TASK}" \
        LIMIT="${EVAL_LIMIT}" \
        IMPORTANCE_PATH="${importance_path}" \
        IMPORTANCE_TAG="${importance_tag}" \
        USE_NEGATED_MODES_STR=1 \
        PRUNE_WHICH_LIST="most,least" \
        MASK_GRANULARITY=kv_group \
        PRUNE_K_FRAC="${PRUNE_FRAC}" \
        LAYER_START="${LAYER_START_DEFAULT}" \
        LAYER_END="${LAYER_END_DEFAULT}" \
        MC_NUM="${mc_num}" \
        bash "${MASK_RUNNER}"
  fi
}

run_adaptive_variant() {
  local path_name="$1"
  local path_mode="$2"
  local run_ts="${RUN_TAG}_${path_name}"
  local importance_path
  importance_path="$(importance_path_for_variant "${path_mode}" "${run_ts}")" || return $?
  local eval_attr_label
  eval_attr_label="$(eval_attr_label_for_dataset "${ATTR_DATASET}")" || return $?
  local item="adaptive_${MODEL_FAMILY}_${path_name}_$(safe_name "${TASK}")"
  local log_path="${STATE_DIR}/${item}.log"
  local importance_tag="pathab_${RUN_TAG}_${MODEL_FAMILY}_${path_name}_$(safe_name "${TASK}")"
  local mc_num=1
  if [ "${TASK}" = "gpqa_main_n_shot" ]; then
    mc_num="${GPQA_MC_NUM:-8}"
  fi

  if [ ! -f "${importance_path}" ]; then
    record_status "${item}" "FAILED" "importance not found: ${importance_path}"
    return 3
  fi

  run_stage "${item}" "${log_path}" \
    env \
      CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      HF_HUB_OFFLINE=1 \
      HF_DATASETS_OFFLINE=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      MODEL_NAME="${MODEL_NAME}" \
      ATTR_METHOD=headig \
      ATTR_DATASETS_STR="${eval_attr_label}" \
      TASKS_STR="${TASK}" \
      LIMIT="${EVAL_LIMIT}" \
      IMPORTANCE_PATH="${importance_path}" \
      IMPORTANCE_TAG="${importance_tag}" \
      USE_NEGATED_MODES_STR=1 \
      MODEL_TYPES_STR=adaptive \
      MC_NUM="${mc_num}" \
      bash "${ADAPTIVE_RUNNER}"
}

model_context || exit $?

echo "========================================================"
echo "Path ablation"
echo "========================================================"
echo "Model family: ${MODEL_FAMILY}"
echo "GPU:          ${GPU_ID}"
echo "Run tag:      ${RUN_TAG}"
echo "Attr dataset: ${ATTR_DATASET} max_samples=${ATTR_MAX_SAMPLES}"
echo "Eval task:    ${TASK} limit=${EVAL_LIMIT}"
echo "IG:           steps=${IG_STEPS} mask_probs=${MASK_PROBS} mcs=${MASK_SAMPLES_PER_PROB}"
echo "Seeds:        seed=${SEED} data=${DATA_SEED} mask=${MASK_SEED} path=${PATH_SEED}"
echo "Prune modes:  most,least"
echo "Adaptive:     ${RUN_ADAPTIVE}"
echo "State dir:    ${STATE_DIR}"
echo "Started:      $(timestamp)"
echo "========================================================"

FAILED_ITEMS=()

PATH_NAMES=(dp stp poly)
PATH_MODES=(diagonal random_threshold random_threshold)
PATH_SAMPLES_LIST=(1 1 4)

for idx in "${!PATH_NAMES[@]}"; do
  name="${PATH_NAMES[$idx]}"
  mode="${PATH_MODES[$idx]}"
  samples="${PATH_SAMPLES_LIST[$idx]}"
  if ! run_attribution_variant "${name}" "${mode}" "${samples}"; then
    FAILED_ITEMS+=("attr:${name}")
    continue
  fi
  imp="$(importance_path_for_variant "${mode}" "${RUN_TAG}_${name}")"
  record_status "importance_${MODEL_FAMILY}_${name}" "PATH" "${imp}"

  if ! run_pruning_variant "${name}" "${mode}"; then
    FAILED_ITEMS+=("mask:${name}")
  fi

  if [ "${RUN_ADAPTIVE}" = "1" ]; then
    if ! run_adaptive_variant "${name}" "${mode}"; then
      FAILED_ITEMS+=("adaptive:${name}")
    fi
  fi
done

if [ "${#FAILED_ITEMS[@]}" -gt 0 ]; then
  record_status "summary" "FAILED" "${FAILED_ITEMS[*]}"
  exit 1
fi

record_status "summary" "DONE" "all path variants completed"
