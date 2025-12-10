#!/usr/bin/env python3
"""
Simple test script to debug adaptive sparse attention
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from transformers import AutoTokenizer, AutoConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

model_path = "/data/qh_models/Dream-v0-Instruct-7B"
device = "cuda"

# Load tokenizer and config
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

n_layers = config.num_hidden_layers
n_heads = config.num_key_value_heads

print(f"Layers: {n_layers}, KV Heads: {n_heads}")

# Create adaptive config
from models.Dream.sparse.adaptive_utils_dream import create_adaptive_sparsity_config

adaptive_config = create_adaptive_sparsity_config(
    n_layers=n_layers,
    n_heads=n_heads,
    strategy='uniform',
    base_sparsity=0.5,
    min_sparsity=0.1,
    max_sparsity=0.9,
    seed=42
)

# Load model
from models.Dream.core.adaptive_sparsed_modeling_dream import AdaptiveDreamModel

print("\nLoading AdaptiveDreamModel...")
model = AdaptiveDreamModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map='auto',
    adaptive_config=adaptive_config
)
model.eval()
print("Model loaded!")

# Test 1: Without SparseD_param (should work like normal model)
print("\n" + "="*70)
print("TEST 1: Without SparseD_param (standard attention)")
print("="*70)

messages = [{"role": "user", "content": "What is the capital of France?"}]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
)
input_ids = inputs.input_ids.to(device)
attention_mask = inputs.attention_mask.to(device)

with torch.no_grad():
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=512,
        output_history=True,
        return_dict_in_generate=True,
        steps=512,
        temperature=0.2,
        top_p=0.95,
        alg="entropy",
        alg_temp=0.,
    )

generations = [
    tokenizer.decode(g[len(p):].tolist())
    for p, g in zip(input_ids, output.sequences)
]
result = generations[0].split(tokenizer.eos_token)[0]
print(f"Input: What is the capital of France?")
print(f"Output: {result}")

# Test 2: With SparseD_param and adaptive=False (sparse but not adaptive)
print("\n" + "="*70)
print("TEST 2: With SparseD_param and adaptive=False")
print("="*70)

from models.Dream.generation_utils.generation_utils_dream import DreamGenerationConfig

generation_config = DreamGenerationConfig(
    max_new_tokens=128,
    steps=128,
    alg='entropy',  # Use entropy like in test_dream.py
    temperature=0.2,  # Use non-zero temperature
    eps=1e-3,
)

SparseD_param = {
    'skip': 0.2,
    'select': 0.3,  # Lower selection ratio
    'block_size': 128,
    'new_generation': 128,
    'whole_steps': 128,
    'adaptive': False,  # Disable adaptive first
}

input_ids = tokenizer("What is the capital of France?", return_tensors="pt").input_ids.to(device)

with torch.no_grad():
    output = model.diffusion_generate(
        inputs=input_ids,
        generation_config=generation_config,
        SparseD_param=SparseD_param
    )

if hasattr(output, 'sequences'):
    output_ids = output.sequences
else:
    output_ids = output

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Input: What is the capital of France?")
print(f"Output: {generated_text}")

# Test 3: With SparseD_param and adaptive=True
print("\n" + "="*70)
print("TEST 3: With SparseD_param and adaptive=True")
print("="*70)

SparseD_param['adaptive'] = True

with torch.no_grad():
    output = model.diffusion_generate(
        inputs=input_ids,
        generation_config=generation_config,
        SparseD_param=SparseD_param
    )

if hasattr(output, 'sequences'):
    output_ids = output.sequences
else:
    output_ids = output

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Input: What is the capital of France?")
print(f"Output: {generated_text}")

print("\n" + "="*70)
print("All tests completed!")
print("="*70)

