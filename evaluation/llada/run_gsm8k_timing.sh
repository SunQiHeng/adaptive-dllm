#!/bin/bash
# 运行 GSM8K 任务并比较三种模式的运行时间
# Usage: bash run_gsm8k_timing.sh

cd /home/qiheng/Projects/adaptive-dllm/evaluation/llada

# 环境设置
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=/home/qiheng/Projects/adaptive-dllm:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=2

# 激活环境
source ~/miniconda3/bin/activate adaptive-dllm

# 创建结果目录
mkdir -p results/standard/gsm8k results/sparse/gsm8k results/adaptive/gsm8k results

# 清空之前的 timing_log
echo "========================================" > results/timing_log.txt
echo "GSM8K Timing Comparison - $(date)" >> results/timing_log.txt
echo "========================================" >> results/timing_log.txt
echo "" >> results/timing_log.txt

echo "=========================================="
echo "开始运行 GSM8K 任务时间对比测试"
echo "任务: GSM8K (20 samples)"
echo "模式: Standard, Sparse, Adaptive"
echo "=========================================="
echo ""

# 运行任务 (GSM8K 是 index 6)
bash run_eval_direct.sh 6

echo ""
echo "=========================================="
echo "所有任务完成！"
echo "=========================================="
echo ""

# 显示运行时间汇总
echo "运行时间汇总:"
echo "=========================================="
if [ -f results/standard/gsm8k/runtime.txt ]; then
    STANDARD_TIME=$(cat results/standard/gsm8k/runtime.txt)
    STANDARD_MIN=$((STANDARD_TIME / 60))
    STANDARD_SEC=$((STANDARD_TIME % 60))
    echo "Standard:  ${STANDARD_MIN}m ${STANDARD_SEC}s (${STANDARD_TIME}s)"
else
    echo "Standard:  未完成"
fi

if [ -f results/sparse/gsm8k/runtime.txt ]; then
    SPARSE_TIME=$(cat results/sparse/gsm8k/runtime.txt)
    SPARSE_MIN=$((SPARSE_TIME / 60))
    SPARSE_SEC=$((SPARSE_TIME % 60))
    echo "Sparse:    ${SPARSE_MIN}m ${SPARSE_SEC}s (${SPARSE_TIME}s)"
else
    echo "Sparse:    未完成"
fi

if [ -f results/adaptive/gsm8k/runtime.txt ]; then
    ADAPTIVE_TIME=$(cat results/adaptive/gsm8k/runtime.txt)
    ADAPTIVE_MIN=$((ADAPTIVE_TIME / 60))
    ADAPTIVE_SEC=$((ADAPTIVE_TIME % 60))
    echo "Adaptive:  ${ADAPTIVE_MIN}m ${ADAPTIVE_SEC}s (${ADAPTIVE_TIME}s)"
else
    echo "Adaptive:  未完成"
fi

echo "=========================================="

# 计算加速比
if [ -f results/standard/gsm8k/runtime.txt ] && [ -f results/sparse/gsm8k/runtime.txt ]; then
    STANDARD_TIME=$(cat results/standard/gsm8k/runtime.txt)
    SPARSE_TIME=$(cat results/sparse/gsm8k/runtime.txt)
    
    if [ "$SPARSE_TIME" -gt 0 ]; then
        SPEEDUP=$(echo "scale=2; ($STANDARD_TIME - $SPARSE_TIME) * 100 / $STANDARD_TIME" | bc)
        echo ""
        echo "Sparse vs Standard:"
        echo "  Speedup: ${SPEEDUP}%"
    fi
fi

if [ -f results/standard/gsm8k/runtime.txt ] && [ -f results/adaptive/gsm8k/runtime.txt ]; then
    STANDARD_TIME=$(cat results/standard/gsm8k/runtime.txt)
    ADAPTIVE_TIME=$(cat results/adaptive/gsm8k/runtime.txt)
    
    if [ "$ADAPTIVE_TIME" -gt 0 ]; then
        SPEEDUP=$(echo "scale=2; ($STANDARD_TIME - $ADAPTIVE_TIME) * 100 / $STANDARD_TIME" | bc)
        echo ""
        echo "Adaptive vs Standard:"
        echo "  Speedup: ${SPEEDUP}%"
    fi
fi

echo ""
echo "详细日志已保存到 results/timing_log.txt"
echo "=========================================="

