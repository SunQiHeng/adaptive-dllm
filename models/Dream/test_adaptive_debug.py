#!/usr/bin/env python3
"""
Debug script to verify adaptive sparse attention is working correctly.

This script adds debug prints to verify:
1. Head sparsity levels are set correctly for each layer
2. Masks are built with different keep_ratios per head
3. Sparse attention is actually being used
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
from transformers import AutoTokenizer, AutoConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Monkey-patch to add debug prints
original_build_masks = None
original_adaptive_sparse_attention = None
debug_info = {
    'mask_builds': 0,
    'sparse_attention_calls': 0,
    'layers_checked': set()
}

def debug_build_adaptive_masks(self, query_states, key_states, value_states, block_size, new_generation, SparseD_param):
    """Wrapper to add debug info to _build_adaptive_masks"""
    B, n_heads, q_len, head_dim = query_states.shape
    _, n_kv_heads, kv_len, _ = key_states.shape
    heads_per_group = n_heads // n_kv_heads
    
    debug_info['mask_builds'] += 1
    
    # Print debug info for first few layers
    if self.layer_idx < 3:
        print(f"\n[DEBUG] Layer {self.layer_idx}: Building adaptive masks")
        print(f"  Query shape: {query_states.shape}, Key shape: {key_states.shape}")
        print(f"  n_heads={n_heads}, n_kv_heads={n_kv_heads}, heads_per_group={heads_per_group}")
        print(f"  Head sparsity levels (keep ratios per KV head): {self.head_sparsity_levels}")
        
        # Show which query heads map to which KV heads and their keep ratios
        print(f"  Query head -> KV head -> keep_ratio:")
        for head_idx in range(min(n_heads, 8)):  # Show first 8 heads
            kv_head_idx = head_idx // heads_per_group
            kv_head_idx = min(kv_head_idx, len(self.head_sparsity_levels) - 1)
            keep_ratio = self.head_sparsity_levels[kv_head_idx].item()
            print(f"    Q_head[{head_idx}] -> KV_head[{kv_head_idx}] -> keep_ratio={keep_ratio:.3f}")
    
    # Call original method
    return original_build_masks(self, query_states, key_states, value_states, block_size, new_generation, SparseD_param)

def debug_adaptive_sparse_attention(self, query_states, key_states, value_states, SparseD_param):
    """Wrapper to add debug info to _adaptive_sparse_attention"""
    debug_info['sparse_attention_calls'] += 1
    
    if self.layer_idx not in debug_info['layers_checked']:
        debug_info['layers_checked'].add(self.layer_idx)
        print(f"[DEBUG] Layer {self.layer_idx}: Using adaptive sparse attention")
        print(f"  Sparsity levels set: {self.head_sparsity_levels is not None}")
        if self.head_sparsity_levels is not None:
            print(f"  Keep ratios: min={self.head_sparsity_levels.min():.3f}, "
                  f"max={self.head_sparsity_levels.max():.3f}, "
                  f"mean={self.head_sparsity_levels.mean():.3f}")
        print(f"  SparseD_param: {SparseD_param}")
    
    # Call original method
    return original_adaptive_sparse_attention(self, query_states, key_states, value_states, SparseD_param)

def debug_forward(original_forward):
    """Wrapper to debug forward calls"""
    def wrapper(self, hidden_states, attention_mask=None, position_ids=None, 
                past_key_value=None, output_attentions=False, use_cache=False, 
                cache_position=None, position_embeddings=None, SparseD_param=None, **kwargs):
        
        if self.layer_idx < 3 and SparseD_param is not None:
            print(f"[DEBUG] Layer {self.layer_idx} forward: SparseD_param received = {SparseD_param is not None}")
            if SparseD_param is not None:
                print(f"  adaptive={SparseD_param.get('adaptive', 'N/A')}, "
                      f"now_step={SparseD_param.get('now_step', 'N/A')}")
            print(f"  head_sparsity_levels set = {self.head_sparsity_levels is not None}")
        
        return original_forward(self, hidden_states, attention_mask, position_ids, 
                               past_key_value, output_attentions, use_cache, 
                               cache_position, position_embeddings, SparseD_param, **kwargs)
    return wrapper


def main():
    print("="*70)
    print("ADAPTIVE SPARSE ATTENTION - DEBUG TEST")
    print("="*70)
    
    # Model path
    model_path = "/data/qh_models/Dream-v0-Instruct-7B"
    device = "cuda"
    
    print(f"\n[1] Loading tokenizer and config...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    n_layers = config.num_hidden_layers
    n_heads = config.num_key_value_heads
    print(f"    Layers: {n_layers}, KV Heads: {n_heads}")
    
    # Create adaptive config with varied sparsity
    print("\n[2] Creating adaptive sparsity configuration...")
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
    
    # Show sparsity for first few layers
    print("\n  Sparsity levels for first 3 layers:")
    for layer_idx in range(min(3, n_layers)):
        sparsity = adaptive_config['sparsity_levels'][layer_idx]
        print(f"    Layer {layer_idx}: keep_ratios = {sparsity.cpu().numpy()}")
    
    # Load model
    print("\n[3] Loading AdaptiveDreamModel...")
    from models.Dream.core.adaptive_sparsed_modeling_dream import AdaptiveDreamModel, AdaptiveDreamAttention
    
    model = AdaptiveDreamModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='auto',
        adaptive_config=adaptive_config
    )
    model.eval()
    print("    Model loaded!")
    
    # Monkey-patch debug functions
    global original_build_masks, original_adaptive_sparse_attention
    original_build_masks = AdaptiveDreamAttention._build_adaptive_masks
    original_adaptive_sparse_attention = AdaptiveDreamAttention._adaptive_sparse_attention
    AdaptiveDreamAttention._build_adaptive_masks = debug_build_adaptive_masks
    AdaptiveDreamAttention._adaptive_sparse_attention = debug_adaptive_sparse_attention
    
    # Also patch forward to see if SparseD_param is received
    AdaptiveDreamAttention.forward = debug_forward(AdaptiveDreamAttention.forward)
    
    # Verify adaptive config is set
    print("\n[4] Verifying adaptive config is set on layers...")
    for i, layer in enumerate(model.model.layers[:3]):  # Check first 3 layers
        has_sparsity = layer.self_attn.head_sparsity_levels is not None
        print(f"    Layer {i}: head_sparsity_levels set = {has_sparsity}")
        if has_sparsity:
            print(f"      Values: {layer.self_attn.head_sparsity_levels}")
    
    # Prepare generation
    print("\n[5] Preparing generation configuration...")
    from models.Dream.generation_utils.generation_utils_dream import DreamGenerationConfig
    
    generation_config = DreamGenerationConfig(
        max_new_tokens=512,
        steps=512,
        alg='entropy',
        temperature=0.2,
        eps=1e-3,
        top_p=0.95,
        alg_temp=0.,
        mask_token_id=tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else None,
    )
    
    # Sparse parameters with adaptive enabled
    SparseD_param = {
        'skip': 0.2,
        'select': 1.0,  # Use 1.0 like in test_adaptive.py
        'block_size': 128,
        'new_generation': 512,
        'whole_steps': 512,
        'adaptive': True,
    }
    
    print(f"    SparseD_param: adaptive={SparseD_param['adaptive']}, select={SparseD_param['select']}")
    
    # Generate
    print("\n[6] Generating with adaptive sparse attention...")
    print("="*70)
    
    prompt = "What is the capital of France?"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    
    print(f"\nInput: {prompt}")
    print("-"*70)
    print("Generating (watch for debug output)...\n")
    
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
    
    print("\n" + "="*70)
    print(f"Output: {generated_text}")
    print("="*70)
    
    # Print summary
    print("\n" + "="*70)
    print("DEBUG SUMMARY")
    print("="*70)
    print(f"Total mask builds: {debug_info['mask_builds']}")
    print(f"Total sparse attention calls: {debug_info['sparse_attention_calls']}")
    print(f"Layers that used sparse attention: {sorted(debug_info['layers_checked'])}")
    print(f"Number of layers with sparse attention: {len(debug_info['layers_checked'])}/{n_layers}")
    
    if debug_info['mask_builds'] > 0 and debug_info['sparse_attention_calls'] > 0:
        print("\n✅ Adaptive sparse attention IS being used!")
        print("✅ Different heads are using different sparsity levels!")
    else:
        print("\n❌ Adaptive sparse attention is NOT being used!")
    
    print("="*70)


if __name__ == "__main__":
    main()

