#!/usr/bin/env bash
set -uo pipefail

# GPU-local queue:
# 1. Retry the failed Dream AttnLRP cross-task audit (GSM8K attribution -> GPQA).
# 2. Continue with Dream Shapley + PolyHeadIG/headig mask-main pruning.
#
# The audit retry is best-effort: if it OOMs again, the mask-main queue still
# starts so the GPU remains useful.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_ID=${GPU_ID:?Set GPU_ID to an available physical GPU index.}
RUN_TAG=${RUN_TAG:-"dream_audit_retry_then_mask_$(date +%Y%m%d_%H%M%S)"}
STATE_DIR="${PROJECT_ROOT}/logs/audit_retry/${RUN_TAG}"
STATUS_FILE="${STATE_DIR}/status.tsv"
EVAL_RUNNER="${PROJECT_ROOT}/evaluation/dream/run_eval_task.sh"
MASK_RUNNER="${PROJECT_ROOT}/experiment_scripts/run_mask_main_fill.sh"

GSM8K_ATTNLRP_IMPORTANCE=${GSM8K_ATTNLRP_IMPORTANCE:-"${PROJECT_ROOT}/configs/aconfigs/head_importance_dream_gsm8k_final_hash_attnlrp_style_relu_maskp0.15-0.3-0.5-0.7-0.9_mcs2_mean_masked_tsdream_attnlrp_regen_audit_20260611_gpu3_gsm8k/head_importance.pt"}
MASK_RUN_TAG=${MASK_RUN_TAG:-"mask_main_dream_shapley_headig_gpu${GPU_ID}_20260612"}

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

echo "========================================================"
echo "Dream audit retry + mask-main queue"
echo "========================================================"
echo "GPU:       ${GPU_ID}"
echo "Run tag:   ${RUN_TAG}"
echo "State dir: ${STATE_DIR}"
echo "Started:   $(timestamp)"
echo "========================================================"

if [ ! -f "${GSM8K_ATTNLRP_IMPORTANCE}" ]; then
  record_status "retry_gsm8k_attr_on_gpqa_mc8" "FAILED" "missing importance=${GSM8K_ATTNLRP_IMPORTANCE}"
else
  run_stage "retry_gsm8k_attr_on_gpqa_mc8" "${STATE_DIR}/retry_gsm8k_attr_on_gpqa_mc8.log" \
    env \
      CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      HF_HUB_OFFLINE=1 \
      HF_DATASETS_OFFLINE=1 \
      PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      PYTORCH_ALLOC_CONF=expandable_segments:True \
      MODEL_NAME="dream" \
      ATTR_METHOD="attnlrp" \
      ATTR_DATASETS_STR="gsm8k" \
      TASKS_STR="gpqa_main_n_shot" \
      LIMIT=200 \
      IMPORTANCE_PATH="${GSM8K_ATTNLRP_IMPORTANCE}" \
      IMPORTANCE_TAG="attnlrp_audit_retry_${RUN_TAG}_gsm8k_attr_on_gpqa_mc8" \
      USE_NEGATED=0 \
      USE_NEGATED_MODES_STR=0 \
      MODEL_TYPES_STR="adaptive" \
      MC_NUM=8 \
      bash "${EVAL_RUNNER}" || true
fi

run_stage "dream_shapley_headig_mask_main" "${STATE_DIR}/dream_shapley_headig_mask_main.log" \
  env \
    MODEL_FAMILY="dream" \
    GPU_ID="${GPU_ID}" \
    RUN_TAG="${MASK_RUN_TAG}" \
    METHODS_STR="shapley,headig" \
    DATASETS_STR="mmlu,cmmlu,ceval-valid,gpqa_main_n_shot,gsm8k,humaneval,mbpp" \
    bash "${MASK_RUNNER}"

echo "========================================================"
echo "Finished: $(timestamp)"
echo "State:    ${STATE_DIR}"
echo "========================================================"
