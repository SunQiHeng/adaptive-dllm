#!/bin/bash
# Quick comparison test with limited examples (no slurm)
# Usage: ./run_compare_quick.sh [task] [limit]
# Example: ./run_compare_quick.sh gsm8k 10

export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=/home/qiheng/Projects/adaptive-dllm:$PYTHONPATH

cd /home/qiheng/Projects/adaptive-dllm/evaluation/llada
mkdir -p logs results/debug

MODEL_PATH="GSAI-ML/LLaDA-8B-Base"
TASK=${1:-"gsm8k"}
LIMIT=${2:-10}
GEN_LENGTH=128
STEPS=128
BLOCK_LENGTH=128

echo "======================================================================="
echo "Quick Comparison Test (Debug Mode)"
echo "======================================================================="
echo "Task: $TASK"
echo "Limit: $LIMIT examples"
echo "Gen Length: $GEN_LENGTH, Steps: $STEPS"
echo "======================================================================="
echo ""

MODEL_TYPES=("standard" "sparse" "adaptive")

for MODEL_TYPE in "${MODEL_TYPES[@]}"; do
    echo ""
    echo "-----------------------------------------------------------------------"
    echo "Testing: $MODEL_TYPE"
    echo "-----------------------------------------------------------------------"
    
    OUTPUT_DIR="results/debug/${TASK}/${MODEL_TYPE}"
    mkdir -p "$OUTPUT_DIR"
    
    START_TIME=$(date +%s)
    
    python eval_llada.py \
        --model llada_eval \
        --model_args model_path="${MODEL_PATH}",model_type="${MODEL_TYPE}",gen_length=${GEN_LENGTH},steps=${STEPS},block_length=${BLOCK_LENGTH},skip=0.2,select=0.3,block_size=128,base_sparsity=0.5 \
        --tasks "${TASK}" \
        --limit ${LIMIT} \
        --output_path "${OUTPUT_DIR}/results.json" \
        --log_samples \
        2>&1 | tee "${OUTPUT_DIR}/eval.log"
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo ""
    echo "✓ Completed $MODEL_TYPE in ${DURATION}s"
    
    # Show quick result
    if [ -f "${OUTPUT_DIR}/results.json" ]; then
        echo "  Quick peek at results:"
        python -c "
import json
try:
    with open('${OUTPUT_DIR}/results.json', 'r') as f:
        data = json.load(f)
    results = data.get('results', {})
    for task, metrics in results.items():
        print(f'    {task}:')
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    print(f'      {k}: {v:.4f}')
except Exception as e:
    print(f'    Error: {e}')
"
    fi
    
    sleep 2
done

echo ""
echo "======================================================================="
echo "Quick test completed!"
echo "======================================================================="
echo ""
echo "Results saved in: results/debug/${TASK}/"
echo ""
echo "View comparison:"
echo "  python compare_results.py --debug ${TASK}"
echo ""
echo "For full evaluation, run:"
echo "  sbatch run_compare_all.slurm ${TASK}"
echo "======================================================================="

