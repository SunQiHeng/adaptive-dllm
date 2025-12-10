#!/bin/bash
# Run all tasks sequentially (bypasses SLURM)
# This will run all 9 tasks one by one, with each task testing 3 model types

cd /home/qiheng/Projects/adaptive-dllm/evaluation/llada

# Create a main log file
MAIN_LOG="logs/run_all_tasks_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "================================================" | tee "$MAIN_LOG"
echo "Starting sequential evaluation of all tasks" | tee -a "$MAIN_LOG"
echo "Started at: $(date)" | tee -a "$MAIN_LOG"
echo "================================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Run tasks 0-8
for i in {0..8}; do
    echo "================================================" | tee -a "$MAIN_LOG"
    echo "Running task $i..." | tee -a "$MAIN_LOG"
    echo "================================================" | tee -a "$MAIN_LOG"
    
    bash run_eval_direct.sh $i 2>&1 | tee -a "$MAIN_LOG"
    
    EXIT_CODE=${PIPESTATUS[0]}
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: Task $i failed with exit code $EXIT_CODE" | tee -a "$MAIN_LOG"
    else
        echo "SUCCESS: Task $i completed" | tee -a "$MAIN_LOG"
    fi
    echo "" | tee -a "$MAIN_LOG"
done

echo "================================================" | tee -a "$MAIN_LOG"
echo "All tasks completed!" | tee -a "$MAIN_LOG"
echo "Finished at: $(date)" | tee -a "$MAIN_LOG"
echo "================================================" | tee -a "$MAIN_LOG"

