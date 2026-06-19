#!/usr/bin/env bash
set -uo pipefail

# Recompute Dream AttnLRP on GPQA/GSM8K, then evaluate both same-task and
# cross-task adaptive sparse inference. This is meant to audit suspicious
# Dream AttnLRP GPQA/GSM8K cells without reusing old attribution files.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID=${GPU_ID:?Set GPU_ID to an available physical GPU index.}
RUN_TAG=${RUN_TAG:-"dream_attnlrp_regen_audit_$(date +%Y%m%d_%H%M%S)"}
STATE_DIR="${PROJECT_ROOT}/logs/attnlrp_audit/${RUN_TAG}"
STATUS_FILE="${STATE_DIR}/status.tsv"
ATTR_RUNNER="${PROJECT_ROOT}/models/Dream/attribution/baseline_attribution/run_attnlrp_head_attribution.sh"
EVAL_RUNNER="${PROJECT_ROOT}/evaluation/dream/run_eval_task.sh"

ATTR_SEED=${ATTR_SEED:-131}
GPQA_MAX_SAMPLES=${GPQA_MAX_SAMPLES:-200}
GSM8K_MAX_SAMPLES=${GSM8K_MAX_SAMPLES:-100}
EVAL_LIMIT_GPQA=${EVAL_LIMIT_GPQA:-200}
EVAL_LIMIT_GSM8K=${EVAL_LIMIT_GSM8K:-200}
GPQA_MC_NUM=${GPQA_MC_NUM:-8}

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

find_regen_importance() {
  local source_prefix="$1"
  local run_ts="$2"
  find "${PROJECT_ROOT}/configs/aconfigs" -mindepth 2 -maxdepth 2 -type f \
    -path "*/head_importance_dream_${source_prefix}_attnlrp_*_ts${run_ts}/head_importance.pt" \
    -printf "%T@ %p\n" \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

run_attnlrp_attribution() {
  local item="$1"
  local dataset="$2"
  local source_prefix="$3"
  local max_samples="$4"
  local run_ts="${RUN_TAG}_${dataset}"
  local log_path="${STATE_DIR}/${item}.log"
  local path_file="${STATE_DIR}/${item}.importance_path"

  if [ -f "${path_file}" ] && [ -f "$(cat "${path_file}")" ]; then
    record_status "${item}" "SKIP" "importance=$(cat "${path_file}")"
    return 0
  fi

  run_stage "${item}" "${log_path}" \
    env \
      GPU_ID="${GPU_ID}" \
      CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      HF_DATASETS_OFFLINE=1 \
      HF_HUB_OFFLINE=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      RUN_TS="${run_ts}" \
      ATTR_DATASETS_STR="${dataset}" \
      MAX_SAMPLES="${max_samples}" \
      GPQA_MAX_SAMPLES="${GPQA_MAX_SAMPLES}" \
      GSM8K_MAX_SAMPLES="${GSM8K_MAX_SAMPLES}" \
      GSM8K_ANSWER_MODE=final_hash \
      SEED="${ATTR_SEED}" \
      DATA_SEED="${ATTR_SEED}" \
      MASK_SEED="${ATTR_SEED}" \
      RELEVANCE_POSTPROCESS=relu \
      MASK_PROBS=0.15,0.3,0.5,0.7,0.9 \
      MASK_SAMPLES_PER_PROB=2 \
      LOSS_NORMALIZE=mean_masked \
      MASK_BATCH_SIZE=1 \
      GRADIENT_CHECKPOINTING=1 \
      bash "${ATTR_RUNNER}"
  local rc=$?
  if [ "${rc}" -ne 0 ]; then
    return "${rc}"
  fi

  local importance_path
  importance_path="$(find_regen_importance "${source_prefix}" "${run_ts}")"
  if [ -z "${importance_path}" ] || [ ! -f "${importance_path}" ]; then
    record_status "${item}" "FAILED" "completed but could not locate regenerated importance source=${source_prefix} ts=${run_ts}"
    return 3
  fi

  echo "${importance_path}" > "${path_file}"
  record_status "${item}" "IMPORTANCE" "${importance_path}"
}

run_dream_adaptive_eval() {
  local item="$1"
  local importance_path="$2"
  local attr_label="$3"
  local task="$4"
  local limit="$5"
  local extra_tag="$6"
  local mc_num="${7:-}"
  local log_path="${STATE_DIR}/${item}.log"

  local env_args=(
    env
    CUDA_VISIBLE_DEVICES="${GPU_ID}"
    HF_DATASETS_OFFLINE=1
    HF_HUB_OFFLINE=1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    MODEL_NAME="dream"
    ATTR_METHOD="attnlrp"
    ATTR_DATASETS_STR="${attr_label}"
    TASKS_STR="${task}"
    LIMIT="${limit}"
    IMPORTANCE_PATH="${importance_path}"
    IMPORTANCE_TAG="attnlrp_audit_${RUN_TAG}_${extra_tag}"
    USE_NEGATED=0
    USE_NEGATED_MODES_STR=0
    MODEL_TYPES_STR="adaptive"
  )
  if [ -n "${mc_num}" ]; then
    env_args+=(MC_NUM="${mc_num}")
  fi

  run_stage "${item}" "${log_path}" "${env_args[@]}" bash "${EVAL_RUNNER}"
}

echo "========================================================"
echo "Dream AttnLRP regeneration audit"
echo "========================================================"
echo "GPU:       ${GPU_ID}"
echo "Run tag:   ${RUN_TAG}"
echo "Seed:      ${ATTR_SEED}"
echo "State dir: ${STATE_DIR}"
echo "Started:   $(timestamp)"
echo "========================================================"

FAILED_ITEMS=()

run_attnlrp_attribution "attr_gpqa" "gpqa_main_n_shot" "gpqa_main_n_shot_all" "${GPQA_MAX_SAMPLES}" || FAILED_ITEMS+=("attr_gpqa")
run_attnlrp_attribution "attr_gsm8k" "gsm8k" "gsm8k_final_hash" "${GSM8K_MAX_SAMPLES}" || FAILED_ITEMS+=("attr_gsm8k")

GPQA_IMPORTANCE=""
GSM8K_IMPORTANCE=""
if [ -f "${STATE_DIR}/attr_gpqa.importance_path" ]; then
  GPQA_IMPORTANCE="$(cat "${STATE_DIR}/attr_gpqa.importance_path")"
fi
if [ -f "${STATE_DIR}/attr_gsm8k.importance_path" ]; then
  GSM8K_IMPORTANCE="$(cat "${STATE_DIR}/attr_gsm8k.importance_path")"
fi

if [ -n "${GPQA_IMPORTANCE}" ] && [ -f "${GPQA_IMPORTANCE}" ]; then
  run_dream_adaptive_eval "eval_gpqa_attr_on_gpqa_mc8" "${GPQA_IMPORTANCE}" "gpqa_main_n_shot_all" "gpqa_main_n_shot" "${EVAL_LIMIT_GPQA}" "gpqa_attr_on_gpqa_mc8" "${GPQA_MC_NUM}" || FAILED_ITEMS+=("eval_gpqa_attr_on_gpqa_mc8")
  run_dream_adaptive_eval "eval_gpqa_attr_on_gsm8k" "${GPQA_IMPORTANCE}" "gpqa_main_n_shot_all" "gsm8k" "${EVAL_LIMIT_GSM8K}" "gpqa_attr_on_gsm8k" || FAILED_ITEMS+=("eval_gpqa_attr_on_gsm8k")
else
  record_status "eval_gpqa_source" "SKIP" "missing regenerated GPQA importance"
fi

if [ -n "${GSM8K_IMPORTANCE}" ] && [ -f "${GSM8K_IMPORTANCE}" ]; then
  run_dream_adaptive_eval "eval_gsm8k_attr_on_gsm8k" "${GSM8K_IMPORTANCE}" "gsm8k" "gsm8k" "${EVAL_LIMIT_GSM8K}" "gsm8k_attr_on_gsm8k" || FAILED_ITEMS+=("eval_gsm8k_attr_on_gsm8k")
  run_dream_adaptive_eval "eval_gsm8k_attr_on_gpqa_mc8" "${GSM8K_IMPORTANCE}" "gsm8k" "gpqa_main_n_shot" "${EVAL_LIMIT_GPQA}" "gsm8k_attr_on_gpqa_mc8" "${GPQA_MC_NUM}" || FAILED_ITEMS+=("eval_gsm8k_attr_on_gpqa_mc8")
else
  record_status "eval_gsm8k_source" "SKIP" "missing regenerated GSM8K importance"
fi

echo "========================================================"
echo "Finished: $(timestamp)"
echo "State:    ${STATE_DIR}"
if [ "${#FAILED_ITEMS[@]}" -gt 0 ]; then
  echo "Failed items: ${FAILED_ITEMS[*]}"
  exit 1
fi
echo "All audit items completed."
echo "========================================================"
