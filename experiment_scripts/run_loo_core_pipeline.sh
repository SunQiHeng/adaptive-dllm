#!/usr/bin/env bash
set -uo pipefail

# Resumable exact-LOO pipeline:
#   attribution -> adaptive sparse eval -> prune-most/prune-least eval
#
# Required:
#   MODEL_FAMILY=llada|dream
#   GPU_ID=<physical GPU index>
#
# Useful overrides:
#   DATASETS_STR="mmlu,cmmlu,ceval-valid,gpqa_main_n_shot,gsm8k,humaneval,mbpp"
#   RUN_TAG=formal_loo_llada_20260606
#   RUN_ADAPTIVE=0
#   RUN_PRUNING=0
#
# Minerva Math is intentionally excluded from the default queue because its
# evaluation protocol and cost are tracked separately in experiment_process.md.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_FAMILY=${MODEL_FAMILY:?Set MODEL_FAMILY=llada or MODEL_FAMILY=dream.}
GPU_ID=${GPU_ID:?Set GPU_ID to an available physical GPU index.}
DATASETS_STR=${DATASETS_STR:-"mmlu,cmmlu,ceval-valid,gpqa_main_n_shot,gsm8k,humaneval,mbpp"}
RUN_TAG=${RUN_TAG:-"formal_loo_${MODEL_FAMILY}_$(date +%Y%m%d_%H%M%S)"}
RUN_ADAPTIVE=${RUN_ADAPTIVE:-1}
RUN_PRUNING=${RUN_PRUNING:-1}

case "${MODEL_FAMILY}" in
  llada)
    MODEL_NAME="llada-1_5"
    ATTR_RUNNER="${PROJECT_ROOT}/models/LLaDA/attribution/baseline_attribution/run_leave_one_out_head_attribution.sh"
    ADAPTIVE_RUNNER="${PROJECT_ROOT}/evaluation/llada/run_eval_task.sh"
    PRUNING_RUNNER="${PROJECT_ROOT}/evaluation/llada/run_eval_mask_head_task.sh"
    LAYER_START=0
    LAYER_END=31
    ;;
  dream)
    MODEL_NAME="dream"
    ATTR_RUNNER="${PROJECT_ROOT}/models/Dream/attribution/baseline_attribution/run_leave_one_out_head_attribution.sh"
    ADAPTIVE_RUNNER="${PROJECT_ROOT}/evaluation/dream/run_eval_task.sh"
    PRUNING_RUNNER="${PROJECT_ROOT}/evaluation/dream/run_eval_mask_head_task.sh"
    LAYER_START=0
    LAYER_END=27
    ;;
  *)
    echo "ERROR: MODEL_FAMILY must be llada or dream; got ${MODEL_FAMILY}" >&2
    exit 2
    ;;
esac

STATE_DIR="${PROJECT_ROOT}/logs/loo_core/${RUN_TAG}"
mkdir -p "${STATE_DIR}"
STATUS_FILE="${STATE_DIR}/status.tsv"
touch "${STATUS_FILE}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

record_status() {
  local dataset="$1"
  local stage="$2"
  local status="$3"
  local detail="${4:-}"
  printf "%s\t%s\t%s\t%s\t%s\n" "$(timestamp)" "${dataset}" "${stage}" "${status}" "${detail}" >> "${STATUS_FILE}"
  echo "[$(timestamp)] dataset=${dataset} stage=${stage} status=${status} ${detail}"
}

eval_attr_dataset() {
  case "$1" in
    mmlu) echo "mmlu_all" ;;
    cmmlu) echo "cmmlu_all" ;;
    ceval-valid) echo "ceval-valid_all" ;;
    gpqa_main_n_shot) echo "gpqa_main_n_shot_all" ;;
    gsm8k) echo "gsm8k" ;;
    minerva_math) echo "minerva_math" ;;
    humaneval) echo "humaneval" ;;
    mbpp) echo "mbpp" ;;
    *)
      echo "ERROR: unsupported attribution dataset: $1" >&2
      return 2
      ;;
  esac
}

run_logged_stage() {
  local log_path="$1"
  shift
  "$@" 2>&1 | tee "${log_path}"
  return "${PIPESTATUS[0]}"
}

find_importance_path() {
  local dataset_run_ts="$1"
  find "${PROJECT_ROOT}/configs/aconfigs" -maxdepth 2 -type f \
    -path "*ts${dataset_run_ts}/head_importance.pt" -print -quit
}

run_attribution() {
  local dataset="$1"
  local dataset_run_ts="$2"
  local log_path="$3"
  run_logged_stage "${log_path}" env \
    GPU_ID="${GPU_ID}" \
    ATTR_DATASETS_STR="${dataset}" \
    RUN_TS="${dataset_run_ts}" \
    LAYER_START="${LAYER_START}" \
    LAYER_END="${LAYER_END}" \
    SCORE_POSTPROCESS="signed" \
    bash "${ATTR_RUNNER}"
}

run_adaptive_eval() {
  local eval_dataset="$1"
  local importance_path="$2"
  local importance_tag="$3"
  local log_path="$4"
  run_logged_stage "${log_path}" env \
    CUDA_VISIBLE_DEVICES="${GPU_ID}" \
    MODEL_NAME="${MODEL_NAME}" \
    ATTR_METHOD="loo" \
    ATTR_DATASETS_STR="${eval_dataset}" \
    IMPORTANCE_PATH="${importance_path}" \
    IMPORTANCE_TAG="${importance_tag}" \
    USE_NEGATED=0 \
    USE_NEGATED_MODES_STR=0 \
    MODEL_TYPES_STR="adaptive" \
    bash "${ADAPTIVE_RUNNER}"
}

run_pruning_eval() {
  local eval_dataset="$1"
  local importance_path="$2"
  local importance_tag="$3"
  local log_path="$4"
  if [ "${MODEL_FAMILY}" = "llada" ]; then
    run_logged_stage "${log_path}" env \
      CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      MODEL_NAME="${MODEL_NAME}" \
      ATTR_METHOD="loo" \
      ATTR_DATASETS_STR="${eval_dataset}" \
      IMPORTANCE_PATH="${importance_path}" \
      IMPORTANCE_TAG="${importance_tag}" \
      USE_NEGATED_MODES_STR=0 \
      MODES="most,least" \
      PRUNE_SCOPE="layer" \
      PRUNE_K_FRAC="0.2" \
      LAYER_START=0 \
      LAYER_END=31 \
      bash "${PRUNING_RUNNER}"
  else
    run_logged_stage "${log_path}" env \
      CUDA_VISIBLE_DEVICES="${GPU_ID}" \
      MODEL_NAME="${MODEL_NAME}" \
      ATTR_METHOD="loo" \
      ATTR_DATASETS_STR="${eval_dataset}" \
      IMPORTANCE_PATH="${importance_path}" \
      IMPORTANCE_TAG="${importance_tag}" \
      USE_NEGATED_MODES_STR=0 \
      PRUNE_WHICH_LIST="most,least" \
      MASK_GRANULARITY="kv_group" \
      PRUNE_K_FRAC="0.05" \
      LAYER_START=0 \
      LAYER_END=27 \
      bash "${PRUNING_RUNNER}"
  fi
}

echo "========================================================"
echo "Exact LOO Core Experiment Pipeline"
echo "========================================================"
echo "Run tag:       ${RUN_TAG}"
echo "Model family:  ${MODEL_FAMILY}"
echo "GPU:           ${GPU_ID}"
echo "Datasets:      ${DATASETS_STR}"
echo "Run adaptive:  ${RUN_ADAPTIVE}"
echo "Run pruning:   ${RUN_PRUNING} (most,least only)"
echo "State dir:     ${STATE_DIR}"
echo "Started:       $(timestamp)"
echo "========================================================"

IFS=',' read -r -a DATASETS <<< "${DATASETS_STR}"
FAILED_DATASETS=()

for raw_dataset in "${DATASETS[@]}"; do
  dataset="$(echo "${raw_dataset}" | xargs)"
  [ -z "${dataset}" ] && continue
  safe_dataset="$(echo "${dataset}" | tr '/-' '__')"
  dataset_dir="${STATE_DIR}/${safe_dataset}"
  mkdir -p "${dataset_dir}"
  dataset_run_ts="${RUN_TAG}_${safe_dataset}"
  importance_path_file="${dataset_dir}/importance_path.txt"
  importance_tag="loo_${MODEL_FAMILY}_${safe_dataset}_${RUN_TAG}"

  if ! eval_dataset="$(eval_attr_dataset "${dataset}")"; then
    record_status "${dataset}" "pipeline" "FAILED" "unsupported dataset"
    FAILED_DATASETS+=("${dataset}")
    continue
  fi

  echo ""
  echo "########################################################"
  echo "Dataset: ${dataset} -> eval attribution label: ${eval_dataset}"
  echo "########################################################"

  importance_path=""
  if [ -f "${dataset_dir}/attribution.done" ] && [ -f "${importance_path_file}" ]; then
    importance_path="$(head -n 1 "${importance_path_file}")"
    if [ -f "${importance_path}" ]; then
      record_status "${dataset}" "attribution" "SKIP" "existing=${importance_path}"
    else
      importance_path=""
      record_status "${dataset}" "attribution" "RETRY" "recorded importance file is missing"
    fi
  fi

  if [ -z "${importance_path}" ]; then
    record_status "${dataset}" "attribution" "RUNNING" "log=${dataset_dir}/attribution.log"
    if run_attribution "${dataset}" "${dataset_run_ts}" "${dataset_dir}/attribution.log"; then
      importance_path="$(find_importance_path "${dataset_run_ts}")"
      if [ -n "${importance_path}" ] && [ -f "${importance_path}" ]; then
        printf "%s\n" "${importance_path}" > "${importance_path_file}"
        touch "${dataset_dir}/attribution.done"
        record_status "${dataset}" "attribution" "DONE" "importance=${importance_path}"
      else
        record_status "${dataset}" "attribution" "FAILED" "head_importance.pt not found"
        FAILED_DATASETS+=("${dataset}")
        continue
      fi
    else
      rc=$?
      record_status "${dataset}" "attribution" "FAILED" "rc=${rc}"
      FAILED_DATASETS+=("${dataset}")
      continue
    fi
  fi

  if [ "${RUN_ADAPTIVE}" = "1" ]; then
    if [ -f "${dataset_dir}/adaptive.done" ]; then
      record_status "${dataset}" "adaptive" "SKIP" "already completed"
    else
      record_status "${dataset}" "adaptive" "RUNNING" "log=${dataset_dir}/adaptive.log"
      if run_adaptive_eval "${eval_dataset}" "${importance_path}" "${importance_tag}" "${dataset_dir}/adaptive.log"; then
        touch "${dataset_dir}/adaptive.done"
        record_status "${dataset}" "adaptive" "DONE"
      else
        rc=$?
        record_status "${dataset}" "adaptive" "FAILED" "rc=${rc}"
      fi
    fi
  fi

  if [ "${RUN_PRUNING}" = "1" ]; then
    if [ -f "${dataset_dir}/pruning.done" ]; then
      record_status "${dataset}" "pruning-most-least" "SKIP" "already completed"
    else
      record_status "${dataset}" "pruning-most-least" "RUNNING" "log=${dataset_dir}/pruning.log"
      if run_pruning_eval "${eval_dataset}" "${importance_path}" "${importance_tag}" "${dataset_dir}/pruning.log"; then
        touch "${dataset_dir}/pruning.done"
        record_status "${dataset}" "pruning-most-least" "DONE"
      else
        rc=$?
        record_status "${dataset}" "pruning-most-least" "FAILED" "rc=${rc}"
      fi
    fi
  fi
done

echo ""
echo "========================================================"
echo "Pipeline finished: $(timestamp)"
echo "State: ${STATE_DIR}"
if [ "${#FAILED_DATASETS[@]}" -gt 0 ]; then
  echo "Attribution failures: ${FAILED_DATASETS[*]}"
  exit 1
fi
echo "All queued attribution datasets completed."
echo "========================================================"
