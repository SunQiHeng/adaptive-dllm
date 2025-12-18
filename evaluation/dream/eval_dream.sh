#!/bin/bash
# Full evaluation script for Dream model
# Supports: standard, sparse, and adaptive sparse attention

# Environment setup
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
export PYTHONPATH=/home/qiheng/Projects/adaptive-dllm:$PYTHONPATH

# Activate environment if needed
# source ~/miniconda3/bin/activate adaptive-dllm

# Model configuration (matching attribution script)
MODEL_PATH="/data/qh_models/Dream-v0-Instruct-7B"
MODEL_TYPE="standard"  # Change to 'sparse' or 'adaptive' as needed

# Generation parameters (matching attribution script)
MAX_NEW_TOKENS=256
STEPS=32
TEMPERATURE=0.8
TOP_P=0.9
ALG="entropy"
ALG_TEMP=0.0

# Sparse parameters (for sparse/adaptive modes)
SKIP=0.2
SELECT=0.3
BLOCK_SIZE=128

# Adaptive parameters (for adaptive mode)
ADAPTIVE_STRATEGY="uniform"  # 'uniform', 'random', or 'normal'
BASE_SPARSITY=0.5
MIN_SPARSITY=0.1
MAX_SPARSITY=0.9
ADAPTIVE_CONFIG_PATH=""  # Optional: path to saved adaptive config

# Multiple choice / likelihood estimation benchmarks
echo "================================================"
echo "Running Multiple Choice Benchmarks"
echo "================================================"

# GPQA (with few-shot)
accelerate launch eval_dream.py \
    --tasks gpqa_main_n_shot \
    --num_fewshot 5 \
    --model dream_eval \
    --batch_size 8 \
    --model_args model_path="${MODEL_PATH}",model_type="${MODEL_TYPE}",mc_num=128,temperature=${TEMPERATURE},top_p=${TOP_P},alg="${ALG}",alg_temp=${ALG_TEMP},skip=${SKIP},select=${SELECT},block_size=${BLOCK_SIZE},adaptive_strategy="${ADAPTIVE_STRATEGY}",base_sparsity=${BASE_SPARSITY},min_sparsity=${MIN_SPARSITY},max_sparsity=${MAX_SPARSITY}

# TruthfulQA
accelerate launch eval_dream.py \
    --tasks truthfulqa_mc2 \
    --num_fewshot 0 \
    --model dream_eval \
    --batch_size 8 \
    --model_args model_path="${MODEL_PATH}",model_type="${MODEL_TYPE}",mc_num=128,temperature=${TEMPERATURE},top_p=${TOP_P},alg="${ALG}",alg_temp=${ALG_TEMP},skip=${SKIP},select=${SELECT},block_size=${BLOCK_SIZE},adaptive_strategy="${ADAPTIVE_STRATEGY}",base_sparsity=${BASE_SPARSITY},min_sparsity=${MIN_SPARSITY},max_sparsity=${MAX_SPARSITY}

# ARC Challenge
accelerate launch eval_dream.py \
    --tasks arc_challenge \
    --num_fewshot 0 \
    --model dream_eval \
    --batch_size 8 \
    --model_args model_path="${MODEL_PATH}",model_type="${MODEL_TYPE}",mc_num=128,temperature=${TEMPERATURE},top_p=${TOP_P},alg="${ALG}",alg_temp=${ALG_TEMP},skip=${SKIP},select=${SELECT},block_size=${BLOCK_SIZE},adaptive_strategy="${ADAPTIVE_STRATEGY}",base_sparsity=${BASE_SPARSITY},min_sparsity=${MIN_SPARSITY},max_sparsity=${MAX_SPARSITY}

# HellaSwag
accelerate launch eval_dream.py \
    --tasks hellaswag \
    --num_fewshot 0 \
    --model dream_eval \
    --batch_size 8 \
    --model_args model_path="${MODEL_PATH}",model_type="${MODEL_TYPE}",mc_num=128,temperature=${TEMPERATURE},top_p=${TOP_P},alg="${ALG}",alg_temp=${ALG_TEMP},skip=${SKIP},select=${SELECT},block_size=${BLOCK_SIZE},adaptive_strategy="${ADAPTIVE_STRATEGY}",base_sparsity=${BASE_SPARSITY},min_sparsity=${MIN_SPARSITY},max_sparsity=${MAX_SPARSITY}

# WinoGrande
accelerate launch eval_dream.py \
    --tasks winogrande \
    --num_fewshot 5 \
    --model dream_eval \
    --batch_size 8 \
    --model_args model_path="${MODEL_PATH}",model_type="${MODEL_TYPE}",mc_num=128,temperature=${TEMPERATURE},top_p=${TOP_P},alg="${ALG}",alg_temp=${ALG_TEMP},skip=${SKIP},select=${SELECT},block_size=${BLOCK_SIZE},adaptive_strategy="${ADAPTIVE_STRATEGY}",base_sparsity=${BASE_SPARSITY},min_sparsity=${MIN_SPARSITY},max_sparsity=${MAX_SPARSITY}

# PIQA
accelerate launch eval_dream.py \
    --tasks piqa \
    --num_fewshot 0 \
    --model dream_eval \
    --batch_size 8 \
    --model_args model_path="${MODEL_PATH}",model_type="${MODEL_TYPE}",mc_num=128,temperature=${TEMPERATURE},top_p=${TOP_P},alg="${ALG}",alg_temp=${ALG_TEMP},skip=${SKIP},select=${SELECT},block_size=${BLOCK_SIZE},adaptive_strategy="${ADAPTIVE_STRATEGY}",base_sparsity=${BASE_SPARSITY},min_sparsity=${MIN_SPARSITY},max_sparsity=${MAX_SPARSITY}

# MMLU
accelerate launch eval_dream.py \
    --tasks mmlu \
    --num_fewshot 5 \
    --model dream_eval \
    --batch_size 1 \
    --model_args model_path="${MODEL_PATH}",model_type="${MODEL_TYPE}",mc_num=1,skip=${SKIP},select=${SELECT},block_size=${BLOCK_SIZE},base_capacity=${BASE_CAPACITY},floor_alpha=${FLOOR_ALPHA}

# CMMLU (Chinese)
accelerate launch eval_dream.py \
    --tasks cmmlu \
    --num_fewshot 5 \
    --model dream_eval \
    --batch_size 1 \
    --model_args model_path="${MODEL_PATH}",model_type="${MODEL_TYPE}",mc_num=1,skip=${SKIP},select=${SELECT},block_size=${BLOCK_SIZE},base_capacity=${BASE_CAPACITY},floor_alpha=${FLOOR_ALPHA}

# CEVAL (Chinese)
accelerate launch eval_dream.py \
    --tasks ceval-valid \
    --num_fewshot 5 \
    --model dream_eval \
    --batch_size 1 \
    --model_args model_path="${MODEL_PATH}",model_type="${MODEL_TYPE}",mc_num=1,skip=${SKIP},select=${SELECT},block_size=${BLOCK_SIZE},base_capacity=${BASE_CAPACITY},floor_alpha=${FLOOR_ALPHA}

echo ""
echo "================================================"
echo "Running Generation Benchmarks"
echo "================================================"

# BBH (Big-Bench Hard)
accelerate launch eval_dream.py \
    --tasks bbh \
    --model dream_eval \
    --model_args model_path="${MODEL_PATH}",model_type="${MODEL_TYPE}",max_new_tokens=${MAX_NEW_TOKENS},steps=${STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},alg="${ALG}",alg_temp=${ALG_TEMP},skip=${SKIP},select=${SELECT},block_size=${BLOCK_SIZE},adaptive_strategy="${ADAPTIVE_STRATEGY}",base_sparsity=${BASE_SPARSITY},min_sparsity=${MIN_SPARSITY},max_sparsity=${MAX_SPARSITY}

# GSM8K (Math)
accelerate launch eval_dream.py \
    --tasks gsm8k \
    --model dream_eval \
    --model_args model_path="${MODEL_PATH}",model_type="${MODEL_TYPE}",max_new_tokens=${MAX_NEW_TOKENS},steps=${STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},alg="${ALG}",alg_temp=${ALG_TEMP},skip=${SKIP},select=${SELECT},block_size=${BLOCK_SIZE},adaptive_strategy="${ADAPTIVE_STRATEGY}",base_sparsity=${BASE_SPARSITY},min_sparsity=${MIN_SPARSITY},max_sparsity=${MAX_SPARSITY}

# Minerva Math
accelerate launch eval_dream.py \
    --tasks minerva_math \
    --model dream_eval \
    --model_args model_path="${MODEL_PATH}",model_type="${MODEL_TYPE}",max_new_tokens=${MAX_NEW_TOKENS},steps=${STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},alg="${ALG}",alg_temp=${ALG_TEMP},skip=${SKIP},select=${SELECT},block_size=${BLOCK_SIZE},adaptive_strategy="${ADAPTIVE_STRATEGY}",base_sparsity=${BASE_SPARSITY},min_sparsity=${MIN_SPARSITY},max_sparsity=${MAX_SPARSITY}

# HumanEval (Code)
accelerate launch eval_dream.py \
    --tasks humaneval \
    --model dream_eval \
    --confirm_run_unsafe_code \
    --model_args model_path="${MODEL_PATH}",model_type="${MODEL_TYPE}",max_new_tokens=${MAX_NEW_TOKENS},steps=${STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},alg="${ALG}",alg_temp=${ALG_TEMP},skip=${SKIP},select=${SELECT},block_size=${BLOCK_SIZE},adaptive_strategy="${ADAPTIVE_STRATEGY}",base_sparsity=${BASE_SPARSITY},min_sparsity=${MIN_SPARSITY},max_sparsity=${MAX_SPARSITY}

# MBPP (Code)
accelerate launch eval_dream.py \
    --tasks mbpp \
    --model dream_eval \
    --confirm_run_unsafe_code \
    --model_args model_path="${MODEL_PATH}",model_type="${MODEL_TYPE}",max_new_tokens=${MAX_NEW_TOKENS},steps=${STEPS},temperature=${TEMPERATURE},top_p=${TOP_P},alg="${ALG}",alg_temp=${ALG_TEMP},skip=${SKIP},select=${SELECT},block_size=${BLOCK_SIZE},adaptive_strategy="${ADAPTIVE_STRATEGY}",base_sparsity=${BASE_SPARSITY},min_sparsity=${MIN_SPARSITY},max_sparsity=${MAX_SPARSITY}

echo ""
echo "================================================"
echo "All evaluations completed!"
echo "================================================"

