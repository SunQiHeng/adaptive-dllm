#!/bin/bash
# Quick debug script for local testing (no slurm)

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=/home/qiheng/Projects/adaptive-dllm:$PYTHONPATH

cd /home/qiheng/Projects/adaptive-dllm/evaluation/llada
mkdir -p logs results

MODEL_PATH="GSAI-ML/LLaDA-8B-Base"
MODEL_TYPE=${1:-"standard"}
TASK=${2:-"gsm8k"}
LIMIT=${3:-10}  # Limit number of examples for quick testing

echo "Debug Mode - Testing with $LIMIT examples"
echo "Model: $MODEL_TYPE, Task: $TASK"

OUTPUT_DIR="results/debug/${MODEL_TYPE}/${TASK}"
mkdir -p "$OUTPUT_DIR"

python eval_llada.py \
    --model llada_eval \
    --model_args model_path="${MODEL_PATH}",model_type="${MODEL_TYPE}",gen_length=128,steps=128,block_length=128,skip=0.2,select=0.3,block_size=128,base_sparsity=0.5 \
    --tasks "${TASK}" \
    --limit ${LIMIT} \
    --output_path "${OUTPUT_DIR}/results.json" \
    --log_samples \
    2>&1 | tee "${OUTPUT_DIR}/eval.log"

echo ""
echo "Debug run completed. Check: $OUTPUT_DIR/results.json"

