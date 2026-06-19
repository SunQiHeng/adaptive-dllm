#!/usr/bin/env bash
set -uo pipefail

# Recompute Dream PolyHeadIG/HeadIG attribution for the suspicious GPQA/GSM8K
# rows, then retest adaptive sparse inference and prune-most with the exact
# newly generated importance files.

PROJECT_ROOT="${PROJECT_ROOT:-/home/qiheng/Projects/adaptive-dllm}"
GPU_ID="${GPU_ID:-5}"
RUN_TAG="${RUN_TAG:-dream_polyig_recheck_$(date +%Y%m%d_%H%M%S)}"
STATE_DIR="${STATE_DIR:-${PROJECT_ROOT}/logs/polyig_recheck/${RUN_TAG}}"
STATUS_FILE="${STATE_DIR}/status.tsv"
PATH_FILE="${STATE_DIR}/importance_paths.tsv"

mkdir -p "${STATE_DIR}"
touch "${STATUS_FILE}" "${PATH_FILE}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" >&2
}

record_status() {
  local status="$1"
  local stage="$2"
  local detail="${3:-}"
  printf '%s\t%s\t%s\t%s\n' "$(date '+%F %T %Z')" "${status}" "${stage}" "${detail}" >> "${STATUS_FILE}"
}

run_stage() {
  local stage="$1"
  shift
  local log_file="${STATE_DIR}/${stage}.log"
  log "START ${stage}"
  record_status "START" "${stage}" "${log_file}"
  "$@" > "${log_file}" 2>&1
  local rc=$?
  if [ "${rc}" -eq 0 ]; then
    log "DONE ${stage}"
    record_status "DONE" "${stage}" "${log_file}"
  else
    log "FAILED ${stage} rc=${rc}"
    record_status "FAILED" "${stage}" "rc=${rc} ${log_file}"
  fi
  return "${rc}"
}

find_importance_path() {
  local source_prefix="$1"
  local run_ts="$2"
  local exact="${PROJECT_ROOT}/configs/aconfigs/head_importance_dream_${source_prefix}_pmrandom_threshold_ts${run_ts}/head_importance.pt"
  if [ -f "${exact}" ]; then
    printf '%s\n' "${exact}"
    return 0
  fi

  find "${PROJECT_ROOT}/configs/aconfigs" \
    -path "*/head_importance_dream_${source_prefix}_pmrandom_threshold_ts${run_ts}/head_importance.pt" \
    -type f | sort | tail -n 1
}

run_polyig_attr() {
  local stage="$1"
  local attr_dataset="$2"
  local source_prefix="$3"
  local max_samples="$4"
  local run_ts="${RUN_TAG}_${source_prefix}"

  if run_stage "attr_${stage}" env \
      GPU_ID="${GPU_ID}" \
      CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      RUN_TS="${run_ts}" \
      ATTR_DATASETS_STR="${attr_dataset}" \
      MAX_SAMPLES="${max_samples}" \
      GPQA_MAX_SAMPLES="${max_samples}" \
      GSM8K_MAX_SAMPLES="${max_samples}" \
      GSM8K_ANSWER_MODE="final_hash" \
      SEED="${ATTR_SEED:-131}" \
      DATA_SEED="${DATA_SEED:-${ATTR_SEED:-131}}" \
      MASK_SEED="${MASK_SEED:-${ATTR_SEED:-131}}" \
      PATH_SEED="${PATH_SEED:-${ATTR_SEED:-131}}" \
      IG_STEPS="${IG_STEPS:-8}" \
      PATH_MODE="random_threshold" \
      PATH_SAMPLES="${PATH_SAMPLES:-4}" \
      MASK_PROBS="${MASK_PROBS:-0.15,0.3,0.5,0.7,0.9}" \
      MASK_SAMPLES_PER_PROB="${MASK_SAMPLES_PER_PROB:-2}" \
      IG_POSTPROCESS="signed" \
      LOSS_NORMALIZE="${LOSS_NORMALIZE:-mean_masked}" \
      MASK_BATCH_SIZE="${MASK_BATCH_SIZE:-1}" \
      GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}" \
      USE_CHAT_TEMPLATE="${USE_CHAT_TEMPLATE:-1}" \
      bash "${PROJECT_ROOT}/models/Dream/attribution/loss_attribution/run_loss_attribution_all_heads.sh"; then
    local importance_path
    importance_path="$(find_importance_path "${source_prefix}" "${run_ts}")"
    if [ -n "${importance_path}" ] && [ -f "${importance_path}" ]; then
      printf '%s\t%s\t%s\t%s\n' "${stage}" "${attr_dataset}" "${source_prefix}" "${importance_path}" >> "${PATH_FILE}"
      record_status "PATH" "${stage}" "${importance_path}"
      printf '%s\n' "${importance_path}"
      return 0
    fi
    record_status "FAILED" "locate_${stage}" "missing importance for ${source_prefix} ${run_ts}"
  fi
  return 1
}

run_adaptive_eval() {
  local stage="$1"
  local attr_label="$2"
  local task="$3"
  local limit="$4"
  local mc_num="$5"
  local importance_path="$6"

  run_stage "adaptive_${stage}" env \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    MODEL_NAME="dream" \
    ATTR_METHOD="headig" \
    ATTR_DATASETS_STR="${attr_label}" \
    TASKS_STR="${task}" \
    MODEL_TYPES_STR="adaptive" \
    LIMIT="${limit}" \
    MC_NUM="${mc_num}" \
    IMPORTANCE_PATH="${importance_path}" \
    IMPORTANCE_TAG="polyig_recheck_${RUN_TAG}_${stage}" \
    USE_NEGATED="1" \
    USE_NEGATED_MODES_STR="1" \
    GQA_WEIGHT_MODE="${GQA_WEIGHT_MODE:-kv}" \
    bash "${PROJECT_ROOT}/evaluation/dream/run_eval_task.sh"
}

run_prune_most_eval() {
  local stage="$1"
  local attr_label="$2"
  local task="$3"
  local limit="$4"
  local mc_num="$5"
  local importance_path="$6"

  run_stage "prune_most_${stage}" env \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    MODEL_NAME="dream" \
    ATTR_METHOD="headig" \
    ATTR_DATASETS_STR="${attr_label}" \
    TASKS_STR="${task}" \
    LIMIT="${limit}" \
    MC_NUM="${mc_num}" \
    IMPORTANCE_PATH="${importance_path}" \
    IMPORTANCE_TAG="polyig_recheck_${RUN_TAG}_${stage}" \
    USE_NEGATED_MODES_STR="1" \
    PRUNE_WHICH_LIST="most" \
    MASK_GRANULARITY="${MASK_GRANULARITY:-kv_group}" \
    PRUNE_K_FRAC="${PRUNE_K_FRAC:-0.05}" \
    LAYER_START="${LAYER_START:-0}" \
    LAYER_END="${LAYER_END:-27}" \
    bash "${PROJECT_ROOT}/evaluation/dream/run_eval_mask_head_task.sh"
}

main() {
  log "Dream PolyIG recheck"
  log "PROJECT_ROOT=${PROJECT_ROOT}"
  log "GPU_ID=${GPU_ID}"
  log "RUN_TAG=${RUN_TAG}"
  log "STATE_DIR=${STATE_DIR}"
  record_status "INFO" "run" "gpu=${GPU_ID} tag=${RUN_TAG}"

  local gpqa_importance=""
  local gsm8k_importance=""

  gpqa_importance="$(run_polyig_attr "gpqa" "gpqa_main_n_shot" "gpqa_main_n_shot_all" "${GPQA_ATTR_SAMPLES:-200}")"
  if [ -n "${gpqa_importance}" ] && [ -f "${gpqa_importance}" ]; then
    run_adaptive_eval "gpqa_to_gpqa" "gpqa_main_n_shot_all" "gpqa_main_n_shot" "${GPQA_EVAL_LIMIT:-200}" "${GPQA_MC_NUM:-8}" "${gpqa_importance}" || true
    run_prune_most_eval "gpqa_to_gpqa" "gpqa_main_n_shot_all" "gpqa_main_n_shot" "${GPQA_EVAL_LIMIT:-200}" "${GPQA_MC_NUM:-8}" "${gpqa_importance}" || true
  else
    record_status "SKIP" "gpqa_eval" "missing gpqa importance"
  fi

  gsm8k_importance="$(run_polyig_attr "gsm8k_final_hash" "gsm8k" "gsm8k_final_hash" "${GSM8K_ATTR_SAMPLES:-100}")"
  if [ -n "${gsm8k_importance}" ] && [ -f "${gsm8k_importance}" ]; then
    run_adaptive_eval "gsm8k_to_gsm8k" "gsm8k" "gsm8k" "${GSM8K_EVAL_LIMIT:-200}" "" "${gsm8k_importance}" || true
    run_prune_most_eval "gsm8k_to_gsm8k" "gsm8k" "gsm8k" "${GSM8K_EVAL_LIMIT:-200}" "" "${gsm8k_importance}" || true
  else
    record_status "SKIP" "gsm8k_eval" "missing gsm8k importance"
  fi

  log "All scheduled stages finished"
  record_status "DONE" "pipeline" "${RUN_TAG}"
}

main "$@"
