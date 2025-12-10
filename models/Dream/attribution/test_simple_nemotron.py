#!/usr/bin/env python3
"""
Simple test for Nemotron attribution without dataset loading
"""
import os
import sys
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from transformers import AutoTokenizer, AutoModel
from models.Dream.attribution.head_attribution_stepwise_dream import StepwiseIntegratedGradientsAttributionDream
from models.Dream.attribution.compute_nemotron_attribution_stepwise_dream import run_generation_with_history, compute_attribution_for_sample

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
print("Testing simple generation...")

model_path = "/data/qh_models/Dream-v0-Instruct-7B"
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
model = model.to("cuda").eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Simple test prompt
prompt = "What is 2+2?"

print(f"\nPrompt: {prompt}")
print("Running generation...")

try:
    history, output, mask_id = run_generation_with_history(
        model, tokenizer, prompt,
        max_new_tokens=32,
        steps=8,
        step_interval=4,
        alg_temp=0.0,  # Use deterministic mode to avoid multinomial issues
        device="cuda"
    )
    
    print(f"Generated: {output}")
    print(f"History length: {len(history)}")
    
    # Compute attribution
    print("\nComputing attribution...")
    attributor = StepwiseIntegratedGradientsAttributionDream(model=model, n_steps=3)
    
    importance_scores = attributor.compute_full_generation_attribution(
        generation_history=history,
        mask_token_id=mask_id,
        attention_mask="full",
        position_ids=None
    )
    
    print("\nAttribution results:")
    for layer_idx in list(importance_scores.keys())[:3]:
        top_heads = importance_scores[layer_idx].topk(5)
        print(f"  Layer {layer_idx}: Top 5 heads = {top_heads.indices.tolist()}")
    
    print("\n✓ Test passed!")
    
except Exception as e:
    print(f"\n✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

