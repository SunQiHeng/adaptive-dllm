#!/usr/bin/env bash
# Formal Minerva Math attribution/evaluation for non-ARA baselines.
# ARA and CoKV are handled by run_matched_ara_cokv_pipeline.sh so that CoKV's
# sampling count can be calibrated against ARA.  This script covers dense,
# AttAttr, AttnLRP, and exact leave-one-out under the same data/mask protocol.

set -uo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/qiheng/Projects/adaptive-dllm}
MODEL_FAMILY=${MODEL_FAMILY:?Set MODEL_FAMILY=llada or dream}
GPU_ID=${GPU_ID:?Set GPU_ID}
RUN_TAG=${RUN_TAG:-minerva_formal_${MODEL_FAMILY}_$(date +%Y%m%d_%H%M%S)}
METHODS_STR=${METHODS_STR:-attarr,attnlrp,loo}
MAX_SAMPLES=${MAX_SAMPLES:-200}
# Attribution still uses MAX_SAMPLES=200.  lm-eval applies its limit to each
# of seven Minerva subtasks, so 29 per subtask yields 203 evaluation examples.
MINERVA_LIMIT_PER_SUBTASK=${MINERVA_LIMIT_PER_SUBTASK:-29}
EVAL_LIMIT=${MINERVA_LIMIT_PER_SUBTASK}
RUN_DENSE=${RUN_DENSE:-1}
RUN_ADAPTIVE=${RUN_ADAPTIVE:-1}
RUN_PRUNING=${RUN_PRUNING:-1}

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PATH="/home/qiheng/miniconda3/envs/adaptive-dllm/bin:$PATH"

case "$MODEL_FAMILY" in
  dream)
    MODEL_NAME=dream
    DATA_SEED=47
    ATTARR_RUNNER="$PROJECT_ROOT/models/Dream/attribution/baseline_attribution/run_attarr_head_attribution.sh"
    ATTNRLP_RUNNER="$PROJECT_ROOT/models/Dream/attribution/baseline_attribution/run_attnlrp_head_attribution.sh"
    LOO_RUNNER="$PROJECT_ROOT/models/Dream/attribution/baseline_attribution/run_leave_one_out_head_attribution.sh"
    EVAL_RUNNER="$PROJECT_ROOT/evaluation/dream/run_eval_task.sh"
    ;;
  llada)
    MODEL_NAME=llada-1_5
    DATA_SEED=123
    ATTARR_RUNNER="$PROJECT_ROOT/models/LLaDA/attribution/baseline_attribution/run_attarr_head_attribution.sh"
    ATTNRLP_RUNNER="$PROJECT_ROOT/models/LLaDA/attribution/baseline_attribution/run_attnlrp_head_attribution.sh"
    LOO_RUNNER="$PROJECT_ROOT/models/LLaDA/attribution/baseline_attribution/run_leave_one_out_head_attribution.sh"
    EVAL_RUNNER="$PROJECT_ROOT/evaluation/llada/run_eval_task.sh"
    ;;
  *) echo "Unsupported MODEL_FAMILY=$MODEL_FAMILY" >&2; exit 2 ;;
esac

STATE_DIR="$PROJECT_ROOT/logs/minerva_formal/$RUN_TAG"
STATUS_FILE="$STATE_DIR/status.tsv"
MANIFEST_FILE="$STATE_DIR/rows_manifest_sha256.txt"
mkdir -p "$STATE_DIR"
touch "$STATUS_FILE"

timestamp() { date '+%F %T'; }
record_status() {
  printf '%s\t%s\t%s\t%s\n' "$(timestamp)" "$1" "$2" "${3:-}" >> "$STATUS_FILE"
  echo "[$(timestamp)] item=$1 status=$2 ${3:-}"
}

run_stage() {
  local item=$1 log=$2
  shift 2
  local marker="$STATE_DIR/${item}.done"
  if [[ -f "$marker" ]]; then
    record_status "$item" SKIP already_completed
    return 0
  fi
  record_status "$item" RUNNING "log=$log"
  "$@" > "$log" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    touch "$marker"
    record_status "$item" DONE
  else
    record_status "$item" FAILED "rc=$rc"
  fi
  return "$rc"
}

runner_for_method() {
  case "$1" in
    attarr) echo "$ATTARR_RUNNER" ;;
    attnlrp) echo "$ATTNRLP_RUNNER" ;;
    loo) echo "$LOO_RUNNER" ;;
    *) echo "Unsupported baseline method=$1" >&2; return 2 ;;
  esac
}

find_importance_path() {
  local method=$1 token=$2
  find "$PROJECT_ROOT/configs/aconfigs" -mindepth 2 -maxdepth 2 -type f \
    -path "*/head_importance_${MODEL_NAME}_minerva_math_${method}_*ts${token}/head_importance.pt" \
    -print -quit
}

validate_importance() {
  local method=$1 path=$2
  local manifest
  manifest=$(python - "$path" "$method" "$MAX_SAMPLES" "$DATA_SEED" <<'PY'
import sys
import torch

path, method, max_samples, seed = sys.argv[1:]
max_samples, seed = int(max_samples), int(seed)
obj = torch.load(path, map_location="cpu")
score = obj.get("importance_scores", obj.get("head_importance"))
if isinstance(score, dict):
    values = [torch.as_tensor(v).reshape(-1).float()
              for _, v in sorted(score.items(), key=lambda item: int(item[0]))]
    score = torch.cat(values) if values else torch.empty(0)
else:
    score = torch.as_tensor(score).reshape(-1).float() if score is not None else torch.empty(0)
if score.numel() == 0 or not bool(torch.isfinite(score).all()):
    raise SystemExit("empty or non-finite attribution scores")
meta = obj.get("metadata", {})
for key, expected in (("max_samples", max_samples), ("rows_loaded", max_samples),
                      ("data_seed", seed), ("mask_seed", seed)):
    if int(meta.get(key, -1)) != expected:
        raise SystemExit(f"metadata mismatch: {key}={meta.get(key)!r}, expected={expected}")
manifest = meta.get("rows_manifest_sha256")
if not manifest:
    raise SystemExit("missing rows_manifest_sha256")
if method == "attarr" and int(meta.get("ig_steps", -1)) != 8:
    raise SystemExit("AttAttr must use eight IG steps")
meta.update({
    "formal_protocol": "minerva_matched_v1",
    "formal_task": "minerva_math",
    "attribution_data_matched": True,
})
obj["metadata"] = meta
torch.save(obj, path)
print(manifest)
PY
  ) || return $?

  if [[ -f "$MANIFEST_FILE" ]]; then
    local expected
    expected=$(tr -d '[:space:]' < "$MANIFEST_FILE")
    if [[ "$manifest" != "$expected" ]]; then
      echo "Manifest mismatch for $method: $manifest != $expected" >&2
      return 4
    fi
  else
    printf '%s\n' "$manifest" > "$MANIFEST_FILE"
  fi
  record_status "validate_${method}" DONE "manifest=$manifest path=$path"
}

run_attribution() {
  local method=$1 token="${RUN_TAG}_${1}" runner path
  runner=$(runner_for_method "$method") || return $?
  path=$(find_importance_path "$method" "$token")
  if [[ -n "$path" && -f "$path" ]]; then
    record_status "attr_${method}" SKIP "existing=$path"
  else
    run_stage "attr_${method}" "$STATE_DIR/attr_${method}.log" env \
      CUDA_VISIBLE_DEVICES="$GPU_ID" GPU_ID="$GPU_ID" \
      ATTR_DATASETS_STR=minerva_math MAX_SAMPLES="$MAX_SAMPLES" RUN_TS="$token" \
      SEED="$DATA_SEED" DATA_SEED="$DATA_SEED" MASK_SEED="$DATA_SEED" \
      MASK_PROBS="0.15,0.3,0.5,0.7,0.9" MASK_SAMPLES_PER_PROB=2 \
      LOSS_NORMALIZE=mean_masked GSM8K_ANSWER_MODE=final_hash \
      IG_STEPS=8 BASELINE=zero IG_POSTPROCESS=signed \
      RELEVANCE_POSTPROCESS=relu SCORE_POSTPROCESS=signed \
      DEBUG_DUMP_SAMPLES=0 SAVE_PER_SAMPLE=0 SHOW_PROGRESS=1 \
      bash "$runner" || return $?
    path=$(find_importance_path "$method" "$token")
  fi
  if [[ -z "$path" || ! -f "$path" ]]; then
    record_status "validate_${method}" FAILED artifact_not_found
    return 3
  fi
  validate_importance "$method" "$path"
}

IFS=',' read -r -a METHODS <<< "$METHODS_STR"
for raw_method in "${METHODS[@]}"; do
  method=$(echo "$raw_method" | xargs)
  [[ -z "$method" ]] && continue
  run_attribution "$method" || exit $?
done

if [[ "$RUN_DENSE" = "1" ]]; then
  run_stage dense "$STATE_DIR/dense.log" env \
    CUDA_VISIBLE_DEVICES="$GPU_ID" MODEL_NAME="$MODEL_NAME" \
    ATTR_METHOD=headig ATTR_DATASETS_STR=minerva_math TASKS_STR=minerva_math \
    LIMIT="$EVAL_LIMIT" IMPORTANCE_TAG="minerva_dense_${RUN_TAG}" \
    USE_NEGATED=0 USE_NEGATED_MODES_STR=0 MODEL_TYPES_STR=standard \
    bash "$EVAL_RUNNER" || exit $?
fi

if [[ "$RUN_ADAPTIVE" = "1" ]]; then
  run_stage adaptive "$STATE_DIR/adaptive.log" env \
    GPU_ID="$GPU_ID" RUN_TAG="${RUN_TAG}_adaptive" ATTR_MAIN_ONLY=1 \
    MODEL_FAMILIES_STR="$MODEL_FAMILY" METHODS_STR="$METHODS_STR" \
    DATASETS_STR=minerva_math \
    bash "$PROJECT_ROOT/experiment_scripts/run_adaptive_table_fill.sh" || exit $?
fi

if [[ "$RUN_PRUNING" = "1" ]]; then
  run_stage pruning "$STATE_DIR/pruning.log" env \
    GPU_ID="$GPU_ID" MODEL_FAMILY="$MODEL_FAMILY" RUN_TAG="${RUN_TAG}_pruning" \
    METHODS_STR="$METHODS_STR" DATASETS_STR=minerva_math PRUNE_MODES=most \
    bash "$PROJECT_ROOT/experiment_scripts/run_mask_main_fill.sh" || exit $?
fi

record_status pipeline DONE "methods=$METHODS_STR max_samples=$MAX_SAMPLES"
