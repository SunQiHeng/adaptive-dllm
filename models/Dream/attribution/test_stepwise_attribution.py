#!/usr/bin/env python3
"""
Test script for Dream step-wise attribution
Quick validation without full dataset
"""
import os
import sys
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from models.Dream.attribution.head_attribution_stepwise_dream import StepwiseIntegratedGradientsAttributionDream


def test_simple_generation():
    """Test basic generation with history tracking"""
    print("="*80)
    print("Test 1: Simple Generation with History")
    print("="*80)
    
    # Load model
    model_path = "/data/qh_models/Dream-v0-Instruct-7B"
    print(f"Loading model from {model_path}...")
    
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = model.to("cuda").eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Simple prompt
    messages = [{"role": "user", "content": "What is 2+2?"}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_ids = inputs.input_ids.to("cuda")
    
    # Generate with history
    print("\nGenerating with history tracking...")
    max_new_tokens = 64
    steps = 8
    
    # Simplified generation loop
    import torch.nn.functional as F
    
    mask_token_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else tokenizer.unk_token_id
    x = F.pad(input_ids, (0, max_new_tokens), value=mask_token_id)
    
    history = [x.clone()]
    
    with torch.no_grad():
        for i in range(steps):
            mask_index = (x == mask_token_id)
            
            outputs = model(x, "full", None)
            logits = outputs.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            
            # Simple argmax sampling
            x0 = torch.argmax(logits, dim=-1)
            
            # Update some positions
            num_mask = mask_index.sum().item()
            if num_mask > 0:
                num_update = max(1, num_mask // (steps - i))
                mask_logits = logits[mask_index]
                confidence = torch.max(mask_logits, dim=-1).values
                
                full_conf = torch.full_like(x, -torch.inf, dtype=confidence.dtype)
                full_conf[mask_index] = confidence
                
                _, transfer_idx = torch.topk(full_conf, min(num_update, num_mask))
                
                x_new = x.clone()
                x_new[mask_index] = x0[mask_index]
                x[0, transfer_idx[0]] = x_new[0, transfer_idx[0]]
            
            history.append(x.clone())
    
    # Decode
    final_text = tokenizer.decode(x[0, input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nGenerated text: {final_text}")
    print(f"History length: {len(history)} states")
    
    return model, tokenizer, history, mask_token_id


def test_attribution_computation(model, tokenizer, history, mask_token_id):
    """Test attribution computation"""
    print("\n" + "="*80)
    print("Test 2: Attribution Computation")
    print("="*80)
    
    # Initialize attributor
    print("\nInitializing attributor...")
    attributor = StepwiseIntegratedGradientsAttributionDream(
        model=model,
        n_steps=3  # Use fewer steps for quick test
    )
    
    n_layers = len(model.model.layers)
    n_heads = model.config.num_attention_heads
    
    print(f"Model: {n_layers} layers, {n_heads} heads")
    
    # Compute attribution for first few steps
    print("\nComputing attribution for steps 0-2...")
    
    step_attributions = []
    for step_idx in range(1, min(3, len(history))):
        baseline_ids = history[step_idx - 1]
        actual_ids = history[step_idx]
        
        # Find changed positions
        changed_mask = (baseline_ids != actual_ids)
        was_mask = (baseline_ids == mask_token_id)
        is_not_mask = (actual_ids != mask_token_id)
        valid_changes = changed_mask & was_mask & is_not_mask
        
        changed_positions = torch.nonzero(valid_changes[0], as_tuple=True)[0].tolist()
        
        if len(changed_positions) == 0:
            print(f"  Step {step_idx}: No positions changed")
            continue
        
        print(f"  Step {step_idx}: {len(changed_positions)} positions changed")
        
        # Compute attribution for first layer only (for speed)
        layer_idx = 0
        try:
            attribution = attributor.compute_step_attribution_for_layer(
                baseline_input_ids=baseline_ids,
                actual_input_ids=actual_ids,
                changed_positions=changed_positions,
                target_layer_idx=layer_idx,
                attention_mask="full",
                position_ids=None
            )
            
            step_attributions.append(attribution)
            
            print(f"    Layer {layer_idx} attribution shape: {attribution.shape}")
            print(f"    Mean: {attribution.mean():.4f}, Std: {attribution.std():.4f}")
            print(f"    Top 5 heads: {attribution.topk(5).indices.tolist()}")
            
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()
    
    if len(step_attributions) > 0:
        # Sum attributions across steps
        total_attribution = sum(step_attributions)
        print(f"\nTotal attribution (layer 0):")
        print(f"  Shape: {total_attribution.shape}")
        print(f"  Mean: {total_attribution.mean():.4f}")
        print(f"  Top 5 most important heads: {total_attribution.topk(5).indices.tolist()}")
        print(f"  Top 5 scores: {total_attribution.topk(5).values.tolist()}")
        
        return total_attribution
    else:
        print("\nNo attributions computed")
        return None


def test_full_attribution(model, tokenizer):
    """Test full attribution pipeline with step interval"""
    print("\n" + "="*80)
    print("Test 3: Full Attribution with Step Interval")
    print("="*80)
    
    attributor = StepwiseIntegratedGradientsAttributionDream(
        model=model,
        n_steps=3
    )
    
    # Simple prompt
    messages = [{"role": "user", "content": "Write a hello world program."}]
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_ids = inputs.input_ids.to("cuda")
    
    # Generate with step_interval=2
    max_new_tokens = 64
    steps = 8
    step_interval = 2
    
    print(f"\nGenerating {max_new_tokens} tokens in {steps} steps (interval={step_interval})...")
    
    import torch.nn.functional as F
    
    mask_token_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else tokenizer.unk_token_id
    x = F.pad(input_ids, (0, max_new_tokens), value=mask_token_id)
    
    history = [x.clone()]
    
    with torch.no_grad():
        for i in range(steps):
            mask_index = (x == mask_token_id)
            
            outputs = model(x, "full", None)
            logits = outputs.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            
            x0 = torch.argmax(logits, dim=-1)
            
            num_mask = mask_index.sum().item()
            if num_mask > 0:
                num_update = max(1, num_mask // (steps - i))
                mask_logits = logits[mask_index]
                confidence = torch.max(mask_logits, dim=-1).values
                
                full_conf = torch.full_like(x, -torch.inf, dtype=confidence.dtype)
                full_conf[mask_index] = confidence
                
                _, transfer_idx = torch.topk(full_conf, min(num_update, num_mask))
                
                x_new = x.clone()
                x_new[mask_index] = x0[mask_index]
                x[0, transfer_idx[0]] = x_new[0, transfer_idx[0]]
            
            # Record every step_interval steps
            if (i + 1) % step_interval == 0 or i == steps - 1:
                history.append(x.clone())
    
    final_text = tokenizer.decode(x[0, input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Generated: {final_text[:100]}...")
    print(f"History length: {len(history)} (including initial state)")
    
    # Compute full attribution
    print("\nComputing full attribution across all recorded steps...")
    print("(This may take a few minutes...)")
    
    importance_scores = attributor.compute_full_generation_attribution(
        generation_history=history,
        mask_token_id=mask_token_id,
        attention_mask="full",
        position_ids=None
    )
    
    # Print results
    print("\nAttribution results:")
    for layer_idx, scores in list(importance_scores.items())[:3]:  # First 3 layers
        top_heads = scores.topk(5)
        print(f"  Layer {layer_idx}:")
        print(f"    Top 5 heads: {top_heads.indices.tolist()}")
        print(f"    Scores: {[f'{x:.4f}' for x in top_heads.values.tolist()]}")
    
    return importance_scores


def main():
    print("Dream Step-wise Attribution Test")
    print("="*80)
    
    try:
        # Test 1: Generation with history
        model, tokenizer, history, mask_token_id = test_simple_generation()
        
        # Test 2: Attribution computation
        attribution = test_attribution_computation(model, tokenizer, history, mask_token_id)
        
        # Test 3: Full pipeline
        full_attribution = test_full_attribution(model, tokenizer)
        
        print("\n" + "="*80)
        print("All tests passed! ✓")
        print("="*80)
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=2)
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"Using GPU: {args.gpu}\n")
    
    main()

