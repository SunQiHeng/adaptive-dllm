#!/usr/bin/env bash
set -uo pipefail

# Formal AttAttr pipeline for the two main intervention tables:
#   missing attribution -> adaptive sparse evaluation -> prune-most/least.
#
# The attribution implementation is the head-level AttAttr adaptation used in
# this repository: element-wise IG on the pre-o_proj attention output, reduced
# to one signed CE-loss attribution per head.  Downstream jobs negate those raw
# scores exactly once so larger means more important.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_FAMILY=${MODEL_FAMILY:?Set MODEL_FAMILY=llada or dream.}
GPU_ID=${GPU_ID:?Set GPU_ID to an available physical GPU index.}
DATASETS_STR=${DATASETS_STR:-"mmlu,cmmlu,ceval-valid,gpqa_main_n_shot,gsm8k,humaneval,mbpp"}
RUN_TAG=${RUN_TAG:-"attarr_main_${MODEL_FAMILY}_$(date +%Y%m%d_%H%M%S)"}
GPQA_MC_NUM=${GPQA_MC_NUM:-8}

# Make validation and all child runners independent of the caller's interactive
# shell initialization.  Individual model runners may activate the environment
# again; doing so is harmless.
CONDA_ENV_BIN="${HOME}/miniconda3/envs/adaptive-dllm/bin"
if [ -d "${CONDA_ENV_BIN}" ]; then
  export PATH="${CONDA_ENV_BIN}:${PATH}"
fi

STATE_DIR="${PROJECT_ROOT}/logs/attarr_main/${RUN_TAG}"
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

case "${MODEL_FAMILY}" in
  llada)
    MODEL_NAME="llada-1_5"
    ATTR_RUNNER="${PROJECT_ROOT}/models/LLaDA/attribution/baseline_attribution/run_attarr_head_attribution.sh"
    ;;
  dream)
    MODEL_NAME="dream"
    ATTR_RUNNER="${PROJECT_ROOT}/models/Dream/attribution/baseline_attribution/run_attarr_head_attribution.sh"
    ;;
  *)
    echo "ERROR: unsupported MODEL_FAMILY=${MODEL_FAMILY}" >&2
    exit 2
    ;;
esac

source_label_for_task() {
  case "$1" in
    mmlu) echo "mmlu_all" ;;
    cmmlu) echo "cmmlu_all" ;;
    ceval-valid) echo "ceval-valid_all" ;;
    gpqa_main_n_shot) echo "gpqa_main_n_shot_all" ;;
    gsm8k) echo "gsm8k_final_hash" ;;
    humaneval) echo "humaneval" ;;
    mbpp) echo "mbpp" ;;
    *)
      echo "ERROR: unsupported task=$1" >&2
      return 2
      ;;
  esac
}

formal_importance_path() {
  local task="$1"
  local source_label
  source_label="$(source_label_for_task "${task}")" || return $?
  local prefix="head_importance_${MODEL_NAME}_${source_label}_attarr_signed_k8_zero_maskp0.15-0.3-0.5-0.7-0.9_mcs2_mean_masked_"
  find "${PROJECT_ROOT}/configs/aconfigs" -mindepth 2 -maxdepth 2 -type f \
    -path "*/${prefix}*/head_importance.pt" \
    ! -path "*/${prefix}*_neg/head_importance.pt" \
    ! -path "*/${prefix}*_neg_neg/head_importance.pt" \
    -printf "%T@ %p\n" \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

validate_importance() {
  local path="$1"
  python - "${path}" <<'PY'
import sys
import torch

obj = torch.load(sys.argv[1], map_location="cpu")
if isinstance(obj, dict) and "importance_scores" in obj:
    scores = obj["importance_scores"]
elif isinstance(obj, dict) and "head_importance" in obj:
    scores = obj["head_importance"]
else:
    scores = obj
if isinstance(scores, dict):
    tensors = [torch.as_tensor(v).reshape(-1).float() for _, v in sorted(scores.items(), key=lambda kv: int(kv[0]))]
    values = torch.cat(tensors)
else:
    values = torch.as_tensor(scores).reshape(-1).float()
if values.numel() == 0 or not bool(torch.isfinite(values).all()):
    raise SystemExit("empty or non-finite head importance")
print(f"validated={sys.argv[1]} scores={values.numel()} min={values.min().item():.6g} max={values.max().item():.6g}")
PY
}

IFS=',' read -r -a DATASETS <<< "${DATASETS_STR}"
FAILED_ATTR=()

echo "========================================================"
echo "Formal AttAttr main-table pipeline"
echo "Model family: ${MODEL_FAMILY}"
echo "GPU:          ${GPU_ID}"
echo "Run tag:      ${RUN_TAG}"
echo "Datasets:     ${DATASETS[*]}"
echo "State dir:    ${STATE_DIR}"
echo "Started:      $(timestamp)"
echo "========================================================"

for raw_task in "${DATASETS[@]}"; do
  task="$(echo "${raw_task}" | xargs)"
  [ -z "${task}" ] && continue
  item="attr_${MODEL_FAMILY}_attarr_$(safe_name "${task}")"
  importance_path="$(formal_importance_path "${task}")"
  if [ -n "${importance_path}" ] && [ -f "${importance_path}" ] && validate_importance "${importance_path}"; then
    record_status "${item}" "SKIP" "formal attribution exists: ${importance_path}"
    continue
  fi

  attr_log="${STATE_DIR}/${item}.log"
  record_status "${item}" "RUNNING" "log=${attr_log}"
  env \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    GPU_ID="${GPU_ID}" \
    ATTR_DATASETS_STR="${task}" \
    RUN_TS="${RUN_TAG}_$(safe_name "${task}")" \
    IG_STEPS=8 \
    MASK_PROBS="0.15,0.3,0.5,0.7,0.9" \
    MASK_SAMPLES_PER_PROB=2 \
    IG_POSTPROCESS=signed \
    bash "${ATTR_RUNNER}" > "${attr_log}" 2>&1
  rc=$?
  importance_path="$(formal_importance_path "${task}")"
  if [ "${rc}" -eq 0 ] && [ -n "${importance_path}" ] && validate_importance "${importance_path}"; then
    record_status "${item}" "DONE" "${importance_path}"
  else
    record_status "${item}" "FAILED" "rc=${rc}"
    FAILED_ATTR+=("${task}")
  fi
done

if [ "${#FAILED_ATTR[@]}" -gt 0 ]; then
  record_status "pipeline_${MODEL_FAMILY}" "FAILED" "attribution failures: ${FAILED_ATTR[*]}"
  exit 1
fi

record_status "adaptive_${MODEL_FAMILY}_attarr" "RUNNING"
env \
  GPU_ID="${GPU_ID}" \
  RUN_TAG="${RUN_TAG}_adaptive" \
  ATTARR_MAIN_ONLY=1 \
  MODEL_FAMILIES_STR="${MODEL_FAMILY}" \
  DATASETS_STR="${DATASETS_STR}" \
  GPQA_MC_NUM="${GPQA_MC_NUM}" \
  bash "${PROJECT_ROOT}/experiment_scripts/run_adaptive_table_fill.sh" \
  > "${STATE_DIR}/adaptive.log" 2>&1
adaptive_rc=$?
if [ "${adaptive_rc}" -eq 0 ]; then
  record_status "adaptive_${MODEL_FAMILY}_attarr" "DONE"
else
  record_status "adaptive_${MODEL_FAMILY}_attarr" "FAILED" "rc=${adaptive_rc}"
fi

record_status "pruning_${MODEL_FAMILY}_attarr" "RUNNING"
env \
  GPU_ID="${GPU_ID}" \
  MODEL_FAMILY="${MODEL_FAMILY}" \
  RUN_TAG="${RUN_TAG}_pruning" \
  METHODS_STR=attarr \
  DATASETS_STR="${DATASETS_STR}" \
  GSM8K_SOURCE_LABELS_STR="gsm8k_final_hash" \
  GPQA_MC_NUM="${GPQA_MC_NUM}" \
  bash "${PROJECT_ROOT}/experiment_scripts/run_mask_main_fill.sh" \
  > "${STATE_DIR}/pruning.log" 2>&1
pruning_rc=$?
if [ "${pruning_rc}" -eq 0 ]; then
  record_status "pruning_${MODEL_FAMILY}_attarr" "DONE"
else
  record_status "pruning_${MODEL_FAMILY}_attarr" "FAILED" "rc=${pruning_rc}"
fi

if [ "${adaptive_rc}" -ne 0 ] || [ "${pruning_rc}" -ne 0 ]; then
  record_status "pipeline_${MODEL_FAMILY}" "FAILED" "adaptive_rc=${adaptive_rc} pruning_rc=${pruning_rc}"
  exit 1
fi

record_status "pipeline_${MODEL_FAMILY}" "DONE"
echo "Finished: $(timestamp)"
