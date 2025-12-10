#!/usr/bin/env python3
"""
Compute Head Attribution for Dream using Llama-Nemotron-Post-Training-Dataset
Extract samples from different categories and compute head importance.

Adapted from LLaDA implementation for Dream model architecture.
"""
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from collections import defaultdict
import json
from tqdm import tqdm
from typing import Dict, List

from models.Dream.attribution.head_attribution_block_dream import BlockwiseIntegratedGradientsAttributionDream


def load_nemotron_samples(samples_per_category: int = 10, seed: int = 42):
    """
    Load samples from Llama-Nemotron-Post-Training-Dataset
    
    Dataset has splits: code, math, science, chat, safety
    
    Args:
        samples_per_category: Number of samples to extract per category
        seed: Random seed for reproducibility
    
    Returns:
        Dict mapping category -> List[sample_dict]
    """
    print("Loading nvidia/Llama-Nemotron-Post-Training-Dataset...")
    
    # Available splits (categories)
    categories = ['code', 'math', 'science', 'chat', 'safety']
    
    np.random.seed(seed)
    sampled_data = {}
    
    for category in categories:
        print(f"\nLoading category: {category}")
        try:
            # Load dataset for this category
            dataset = load_dataset(
                "nvidia/Llama-Nemotron-Post-Training-Dataset", 
                split=category,
                streaming=True
            )
            
            # Collect samples
            samples = []
            for i, sample in enumerate(dataset):
                samples.append(sample)
                if len(samples) >= samples_per_category * 2:  # Get more to sample from
                    break
            
            # Random sample
            if len(samples) > samples_per_category:
                indices = np.random.choice(len(samples), samples_per_category, replace=False)
                samples = [samples[i] for i in indices]
            
            sampled_data[category] = samples
            print(f"  ✓ Sampled {len(samples)} samples")
            
        except Exception as e:
            print(f"  ✗ Error loading {category}: {e}")
            continue
    
    return sampled_data


def prepare_prompt(sample: Dict) -> str:
    """
    Prepare prompt from Nemotron sample
    
    Nemotron format: 
    - input: List[Dict] with 'role' and 'content' keys
    - output: String (assistant response)
    """
    prompt_parts = []
    
    # Handle input (conversation history)
    if 'input' in sample:
        input_data = sample['input']
        
        # Input can be a list of messages or a string
        if isinstance(input_data, list):
            for msg in input_data:
                if isinstance(msg, dict):
                    role = msg.get('role', '')
                    content = msg.get('content', '')
                    if role and content:
                        prompt_parts.append(f"{role}: {content}")
        elif isinstance(input_data, str):
            prompt_parts.append(input_data)
    
    # Optionally include system prompt
    if 'system_prompt' in sample and sample['system_prompt']:
        system = sample['system_prompt']
        if isinstance(system, str) and len(system) > 0:
            prompt_parts.insert(0, f"System: {system}")
    
    # Join all parts
    prompt = "\n\n".join(prompt_parts)
    
    # Truncate if too long (keep first part for context)
    if len(prompt) > 2048:
        prompt = prompt[:2048] + "..."
    
    return prompt if prompt else "Hello"


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None):
    """Sample tokens from logits (simplified version for Dream)."""
    if temperature > 0:
        logits = logits / temperature
    
    probs = torch.softmax(logits, dim=-1)
    
    if temperature > 0:
        confidence, x0 = probs.max(dim=-1)
        # Could use torch.multinomial for stochastic sampling
    else:
        confidence, x0 = probs.max(dim=-1)
    
    return confidence, x0


def run_generation_with_blocks(
    model,
    tokenizer,
    prompt: str,
    gen_length: int = 256,
    steps: int = 256,
    block_size: int = 32,
    mask_token_id: int = None,
    device: str = "cuda"
):
    """
    Run Dream diffusion generation and capture block states
    
    Returns:
        blocks_states: List[Dict] with baseline_ids, actual_ids, block_positions
        generated_ids: Final generated sequence
    """
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    prompt_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Get mask token id
    if mask_token_id is None:
        mask_token_id = tokenizer.mask_token_id
        if mask_token_id is None:
            raise ValueError("Model must have a mask token for diffusion generation")
    
    # Initialize sequence
    x = torch.full(
        (prompt_ids.shape[0], prompt_ids.shape[1] + gen_length),
        mask_token_id,
        dtype=torch.long
    ).to(device)
    x[:, :prompt_ids.shape[1]] = prompt_ids.clone()
    
    # Extend attention mask
    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((prompt_ids.shape[0], gen_length), 
                      dtype=attention_mask.dtype, device=device)
        ], dim=-1)
    else:
        attention_mask = "full"
    
    prompt_index = (x != mask_token_id)
    
    # Block-wise generation
    assert gen_length % block_size == 0
    num_blocks = gen_length // block_size
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    
    blocks_states = []
    
    # Prepare generation config for Dream
    from models.Dream.generation.generation_dream import DreamGenerationConfig
    
    for num_block in range(num_blocks):
        block_start = prompt_ids.shape[1] + num_block * block_size
        block_end = prompt_ids.shape[1] + (num_block + 1) * block_size
        
        # Save baseline (block start)
        baseline_ids = x.clone()
        
        # Generate this block using Dream's diffusion process
        # For each step within the block
        timesteps = torch.linspace(1, 1e-3, steps_per_block + 1, device=device)
        
        for i in range(steps_per_block):
            mask_index = (x == mask_token_id)
            
            with torch.no_grad():
                outputs = model(x, attention_mask=attention_mask)
                logits = outputs.logits
                # Dream uses shifted logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            
            # Sample tokens
            mask_logits = logits[mask_index]
            confidence, x0_masked = sample_tokens(mask_logits, temperature=0.0)
            
            # Apply diffusion schedule
            t = timesteps[i]
            s = timesteps[i + 1]
            p_transfer = 1 - s / t if i < steps_per_block - 1 else 1
            
            # Update masked positions
            x0 = torch.zeros_like(x[mask_index], device=device, dtype=torch.long) + mask_token_id
            transfer_index_t_s = torch.rand(*x0.shape, device=device) < p_transfer
            x0[transfer_index_t_s] = x0_masked[transfer_index_t_s]
            x[mask_index] = x0.clone()
        
        # Save actual (block end)
        actual_ids = x.clone()
        
        blocks_states.append({
            'baseline_ids': baseline_ids,
            'actual_ids': actual_ids,
            'block_positions': list(range(block_start, block_end)),
            'block_idx': num_block
        })
    
    return blocks_states, x


def compute_attribution_for_sample(
    model,
    tokenizer,
    sample_text: str,
    category: str,
    gen_length: int = 256,
    steps: int = 256,
    block_size: int = 32,
    n_attribution_steps: int = 10,
    device: str = "cuda"
):
    """
    Compute head attribution for a single sample
    
    Returns:
        attribution_scores: shape (n_layers, n_heads)
    """
    print(f"\n{'='*80}")
    print(f"Category: {category}")
    print(f"Prompt: {sample_text[:100]}...")
    print(f"{'='*80}\n")
    
    # Run generation and get block states
    blocks_states, generated_ids = run_generation_with_blocks(
        model, tokenizer, sample_text,
        gen_length=gen_length,
        steps=steps,
        block_size=block_size,
        device=device
    )
    
    # Initialize attribution computer
    attributor = BlockwiseIntegratedGradientsAttributionDream(
        model=model,
        n_steps=n_attribution_steps
    )
    
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    
    # Accumulate attribution scores across all blocks
    layer_head_scores = torch.zeros(n_layers, n_heads, device=device)
    
    print(f"Computing attribution for {len(blocks_states)} blocks...")
    
    for block_state in tqdm(blocks_states, desc="Blocks"):
        baseline_ids = block_state['baseline_ids']
        actual_ids = block_state['actual_ids']
        block_positions = block_state['block_positions']
        
        # Compute attribution for each layer
        for layer_idx in range(n_layers):
            try:
                attribution = attributor.compute_block_attribution_for_layer(
                    baseline_input_ids=baseline_ids,
                    actual_input_ids=actual_ids,
                    block_positions=block_positions,
                    target_layer_idx=layer_idx,
                    attention_mask="full",
                    position_ids=None
                )
                
                layer_head_scores[layer_idx] += attribution.to(device)
            
            except Exception as e:
                print(f"Error at layer {layer_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Normalize by number of blocks
    layer_head_scores /= len(blocks_states)
    
    return layer_head_scores.cpu().numpy()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to Dream model')
    parser.add_argument('--samples_per_category', type=int, default=5)
    parser.add_argument('--gen_length', type=int, default=256)
    parser.add_argument('--steps', type=int, default=256)
    parser.add_argument('--block_size', type=int, default=32)
    parser.add_argument('--n_attribution_steps', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='./attribution_results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("Head Attribution for Dream on Llama-Nemotron Dataset")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Samples per category: {args.samples_per_category}")
    print(f"Generation length: {args.gen_length}")
    print(f"Steps: {args.steps}")
    print(f"Block size: {args.block_size}")
    print(f"Attribution steps: {args.n_attribution_steps}")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForMaskedLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='auto'
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Load dataset samples
    print("\nLoading dataset samples...")
    sampled_data = load_nemotron_samples(
        samples_per_category=args.samples_per_category,
        seed=args.seed
    )
    
    # Store all results
    all_results = {}
    
    # Process each category
    for category, samples in sampled_data.items():
        print(f"\n{'#'*80}")
        print(f"Processing category: {category} ({len(samples)} samples)")
        print(f"{'#'*80}")
        
        category_attributions = []
        
        for sample_idx, sample in enumerate(samples):
            print(f"\nSample {sample_idx + 1}/{len(samples)}")
            
            # Prepare prompt
            prompt = prepare_prompt(sample)
            
            if len(prompt) < 10:
                print("  Skipping: prompt too short")
                continue
            
            # Compute attribution
            try:
                attribution_scores = compute_attribution_for_sample(
                    model=model,
                    tokenizer=tokenizer,
                    sample_text=prompt,
                    category=category,
                    gen_length=args.gen_length,
                    steps=args.steps,
                    block_size=args.block_size,
                    n_attribution_steps=args.n_attribution_steps,
                    device=args.device
                )
                
                category_attributions.append(attribution_scores)
                
                print(f"  Attribution shape: {attribution_scores.shape}")
                print(f"  Mean score: {attribution_scores.mean():.4f}")
                print(f"  Std score: {attribution_scores.std():.4f}")
                
            except Exception as e:
                print(f"  Error: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if len(category_attributions) > 0:
            # Average across samples
            avg_attribution = np.stack(category_attributions).mean(axis=0)
            all_results[category] = {
                'attribution_scores': avg_attribution.tolist(),
                'n_samples': len(category_attributions),
                'shape': list(avg_attribution.shape)
            }
            
            # Save category result
            output_file = os.path.join(args.output_dir, f'attribution_{category}.npy')
            np.save(output_file, avg_attribution)
            print(f"\nSaved {category} attribution to {output_file}")
    
    # Save summary
    summary_file = os.path.join(args.output_dir, 'attribution_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'categories': list(all_results.keys()),
            'model': args.model_path,
            'config': vars(args)
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Attribution computation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Categories processed: {list(all_results.keys())}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

