#!/usr/bin/env bash
set -uo pipefail

# Queue matched path-design ablations over multiple tasks on one GPU.
# Each task delegates to run_path_ablation_mmlu.sh, which runs DP/STP/PolyHeadIG
# attribution plus pruning and optional adaptive sparse evaluation.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_FAMILY=${MODEL_FAMILY:?Set MODEL_FAMILY=llada or dream.}
GPU_ID=${GPU_ID:?Set GPU_ID to an available physical GPU index.}
RUN_TAG_BASE=${RUN_TAG_BASE:-"path_ablation_${MODEL_FAMILY}_multitask_$(date +%Y%m%d_%H%M%S)"}
TASKS_STR=${TASKS_STR:-"cmmlu,gsm8k,gpqa_main_n_shot"}
RUN_ADAPTIVE=${RUN_ADAPTIVE:-1}
SEED=${SEED:-123}
DATA_SEED=${DATA_SEED:-${SEED}}
MASK_SEED=${MASK_SEED:-${SEED}}
PATH_SEED=${PATH_SEED:-${SEED}}
IG_STEPS=${IG_STEPS:-8}
MASK_PROBS=${MASK_PROBS:-"0.15,0.3,0.5,0.7,0.9"}
MASK_SAMPLES_PER_PROB=${MASK_SAMPLES_PER_PROB:-2}
GPQA_MC_NUM=${GPQA_MC_NUM:-8}

STATE_DIR="${PROJECT_ROOT}/logs/path_ablation/${RUN_TAG_BASE}"
STATUS_FILE="${STATE_DIR}/multitask_status.tsv"
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

default_attr_samples() {
  case "$1" in
    cmmlu) echo "${CMMLU_ATTR_SAMPLES:-60}" ;;
    gsm8k) echo "${GSM8K_ATTR_SAMPLES:-80}" ;;
    gpqa_main_n_shot) echo "${GPQA_ATTR_SAMPLES:-80}" ;;
    ceval-valid) echo "${CEVAL_ATTR_SAMPLES:-60}" ;;
    mmlu) echo "${MMLU_ATTR_SAMPLES:-40}" ;;
    *) echo "${DEFAULT_ATTR_SAMPLES:-60}" ;;
  esac
}

default_eval_limit() {
  case "$1" in
    cmmlu) echo "${CMMLU_EVAL_LIMIT:-80}" ;;
    gsm8k) echo "${GSM8K_EVAL_LIMIT:-100}" ;;
    gpqa_main_n_shot) echo "${GPQA_EVAL_LIMIT:-120}" ;;
    ceval-valid) echo "${CEVAL_EVAL_LIMIT:-100}" ;;
    mmlu) echo "${MMLU_EVAL_LIMIT:-40}" ;;
    *) echo "${DEFAULT_EVAL_LIMIT:-80}" ;;
  esac
}

echo "========================================================"
echo "Multi-task path ablation"
echo "========================================================"
echo "Model family: ${MODEL_FAMILY}"
echo "GPU:          ${GPU_ID}"
echo "Run tag base: ${RUN_TAG_BASE}"
echo "Tasks:        ${TASKS_STR}"
echo "Adaptive:     ${RUN_ADAPTIVE}"
echo "Seeds:        seed=${SEED} data=${DATA_SEED} mask=${MASK_SEED} path=${PATH_SEED}"
echo "State dir:    ${STATE_DIR}"
echo "Started:      $(timestamp)"
echo "========================================================"

record_status "pipeline" "START" "tasks=${TASKS_STR} gpu=${GPU_ID}"

IFS=',' read -r -a TASKS <<< "${TASKS_STR}"
FAILED_ITEMS=()

for raw_task in "${TASKS[@]}"; do
  task="$(echo "${raw_task}" | xargs)"
  [ -n "${task}" ] || continue
  task_tag="$(safe_name "${task}")"
  run_tag="${RUN_TAG_BASE}_${task_tag}"
  attr_samples="$(default_attr_samples "${task}")"
  eval_limit="$(default_eval_limit "${task}")"
  log_path="${STATE_DIR}/${task_tag}.pipeline.log"

  record_status "${task}" "RUNNING" "run_tag=${run_tag} attr_samples=${attr_samples} eval_limit=${eval_limit} log=${log_path}"

  env \
    MODEL_FAMILY="${MODEL_FAMILY}" \
    GPU_ID="${GPU_ID}" \
    RUN_TAG="${run_tag}" \
    ATTR_DATASET="${task}" \
    TASK="${task}" \
    ATTR_MAX_SAMPLES="${attr_samples}" \
    EVAL_LIMIT="${eval_limit}" \
    RUN_ADAPTIVE="${RUN_ADAPTIVE}" \
    SEED="${SEED}" \
    DATA_SEED="${DATA_SEED}" \
    MASK_SEED="${MASK_SEED}" \
    PATH_SEED="${PATH_SEED}" \
    IG_STEPS="${IG_STEPS}" \
    MASK_PROBS="${MASK_PROBS}" \
    MASK_SAMPLES_PER_PROB="${MASK_SAMPLES_PER_PROB}" \
    GPQA_MC_NUM="${GPQA_MC_NUM}" \
    bash "${PROJECT_ROOT}/experiment_scripts/run_path_ablation_mmlu.sh" > "${log_path}" 2>&1
  rc=$?

  if [ "${rc}" -eq 0 ]; then
    record_status "${task}" "DONE" "run_tag=${run_tag}"
  else
    record_status "${task}" "FAILED" "rc=${rc} run_tag=${run_tag}"
    FAILED_ITEMS+=("${task}:rc=${rc}")
  fi
done

if [ "${#FAILED_ITEMS[@]}" -gt 0 ]; then
  record_status "pipeline" "FAILED" "${FAILED_ITEMS[*]}"
  exit 1
fi

record_status "pipeline" "DONE" "all tasks completed"
