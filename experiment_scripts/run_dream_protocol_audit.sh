#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID=${GPU_ID:?Set GPU_ID to an available physical GPU index.}
RUN_TAG=${RUN_TAG:-"dream_protocol_audit_$(date +%Y%m%d_%H%M%S)"}
STATE_DIR="${PROJECT_ROOT}/logs/protocol_audit/${RUN_TAG}"
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

EVAL_RUNNER="${PROJECT_ROOT}/evaluation/dream/run_eval_task.sh"
LOO_GPQA_IMPORTANCE="${PROJECT_ROOT}/configs/aconfigs/head_importance_dream_gpqa_main_n_shot_all_loo_signed_maskp0.15-0.3-0.5-0.7-0.9_mcs2_mean_masked_tsformal_loo_dream_gpu1_20260607_gpqa_main_n_shot/head_importance.pt"

echo "========================================================"
echo "Dream protocol audit"
echo "========================================================"
echo "GPU:       ${GPU_ID}"
echo "Run tag:   ${RUN_TAG}"
echo "State dir: ${STATE_DIR}"
echo "Started:   $(timestamp)"
echo "========================================================"

FAILED_ITEMS=()

run_stage "gpqa_dense_mc8" "${STATE_DIR}/gpqa_dense_mc8.log" env \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  MODEL_NAME="dream" \
  ATTR_METHOD="headig" \
  ATTR_DATASETS_STR="gpqa_main_n_shot_all" \
  TASKS_STR="gpqa_main_n_shot" \
  LIMIT=200 \
  MC_NUM=8 \
  IMPORTANCE_TAG="audit_${RUN_TAG}_gpqa_dense_mc8" \
  USE_NEGATED=0 \
  USE_NEGATED_MODES_STR=0 \
  MODEL_TYPES_STR="standard" \
  bash "${EVAL_RUNNER}" || FAILED_ITEMS+=("gpqa_dense_mc8")

run_stage "gpqa_loo_adaptive_mc8" "${STATE_DIR}/gpqa_loo_adaptive_mc8.log" env \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  MODEL_NAME="dream" \
  ATTR_METHOD="loo" \
  ATTR_DATASETS_STR="gpqa_main_n_shot_all" \
  TASKS_STR="gpqa_main_n_shot" \
  LIMIT=200 \
  MC_NUM=8 \
  IMPORTANCE_PATH="${LOO_GPQA_IMPORTANCE}" \
  IMPORTANCE_TAG="audit_${RUN_TAG}_gpqa_loo_adaptive_mc8" \
  USE_NEGATED=0 \
  USE_NEGATED_MODES_STR=0 \
  MODEL_TYPES_STR="adaptive" \
  bash "${EVAL_RUNNER}" || FAILED_ITEMS+=("gpqa_loo_adaptive_mc8")

run_stage "mbpp_dense_postprocess_l20" "${STATE_DIR}/mbpp_dense_postprocess_l20.log" env \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  MODEL_NAME="dream" \
  ATTR_METHOD="headig" \
  ATTR_DATASETS_STR="mbpp" \
  TASKS_STR="mbpp" \
  LIMIT=20 \
  IMPORTANCE_TAG="audit_${RUN_TAG}_mbpp_dense_postprocess_l20" \
  USE_NEGATED=0 \
  USE_NEGATED_MODES_STR=0 \
  MODEL_TYPES_STR="standard" \
  bash "${EVAL_RUNNER}" || FAILED_ITEMS+=("mbpp_dense_postprocess_l20")

echo "========================================================"
echo "Finished: $(timestamp)"
echo "State:    ${STATE_DIR}"
if [ "${#FAILED_ITEMS[@]}" -gt 0 ]; then
  echo "Failed items: ${FAILED_ITEMS[*]}"
  exit 1
fi
echo "All audit items completed."
echo "========================================================"
