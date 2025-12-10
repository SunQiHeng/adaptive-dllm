#!/usr/bin/env python3
"""
Compute Head Attribution for Dream using Llama-Nemotron-Post-Training-Dataset
Using step-wise attribution adapted for Dream's parallel refinement mechanism.
"""
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from collections import defaultdict
import json
from tqdm import tqdm
from typing import Dict, List, Optional

from models.Dream.attribution.head_attribution_stepwise_dream import StepwiseIntegratedGradientsAttributionDream


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


def prepare_chat_prompt(sample: Dict, tokenizer) -> str:
    """
    Prepare chat-formatted prompt for Dream model
    """
    messages = []
    
    # Handle input
    if 'input' in sample:
        input_data = sample['input']
        if isinstance(input_data, list):
            for msg in input_data:
                if isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    if content:
                        messages.append({"role": role, "content": content})
        elif isinstance(input_data, str):
            messages.append({"role": "user", "content": input_data})
    
    if not messages:
        messages = [{"role": "user", "content": "Hello"}]
    
    # Apply chat template if available
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted_prompt
    except:
        # Fallback to simple format
        return prepare_prompt(sample)


def sample_tokens(logits, temperature=1.0, top_p=0.95, top_k=-1, neg_entropy=False, margin_confidence=False):
    """
    Sample tokens from logits with various strategies
    
    Returns:
        confidence: confidence scores for each position
        sampled_tokens: sampled token ids
    """
    if temperature == 0:
        confidence, sampled_tokens = torch.max(logits, dim=-1)
        return confidence, sampled_tokens
    
    # Apply temperature
    logits = logits / temperature
    
    # Apply top-k filtering
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')
    
    # Apply top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 0] = False
        
        # Scatter sorted tensors back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = -float('Inf')
    
    # Compute probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Sample
    sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    # Compute confidence
    if neg_entropy:
        # Negative entropy as confidence
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        confidence = -entropy
    elif margin_confidence:
        # Margin between top 2 probabilities
        top2_probs = torch.topk(probs, k=2, dim=-1).values
        confidence = top2_probs[..., 0] - top2_probs[..., 1]
    else:
        # Default: max probability
        confidence = probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
    
    return confidence, sampled_tokens


def run_generation_with_history(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    steps: int = 32,
    temperature: float = 0.8,
    top_p: float = 0.9,
    alg: str = "entropy",
    alg_temp: float = 1.5,
    device: str = "cuda",
    step_interval: int = 1,  # 每隔几步记录一次状态
):
    """
    Run Dream diffusion generation and capture step-wise history
    
    Args:
        step_interval: Record state every N steps (default: 1 = every step)
    
    Returns:
        generation_history: List of sequence states at each recorded step
        final_output: Final generated text
    """
    # Prepare input
    messages = [{"role": "user", "content": prompt}]
    
    try:
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
    except:
        inputs = tokenizer(prompt, return_tensors="pt", return_dict=True)
    
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device) if 'attention_mask' in inputs else None
    
    # Get mask token id
    mask_token_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else tokenizer.unk_token_id
    
    # Initialize sequence with masks
    prompt_length = input_ids.shape[1]
    max_length = prompt_length + max_new_tokens
    
    x = F.pad(input_ids, (0, max_new_tokens), value=mask_token_id)
    
    if attention_mask is not None:
        attention_mask = F.pad(attention_mask, (0, max_new_tokens), value=1.0)
        attention_mask = "full"  # Dream uses "full" for causal mask
    else:
        attention_mask = "full"
    
    # Position ids
    tok_idx = None
    
    # Timesteps for diffusion
    eps = 0.001
    timesteps = torch.linspace(1, eps, steps + 1, device=device)
    
    # Record initial state
    generation_history = [x.clone()]
    
    # Diffusion loop
    with torch.no_grad():
        for i in range(steps):
            mask_index = (x == mask_token_id)
            
            # Forward pass
            outputs = model(x, attention_mask, tok_idx)
            logits = outputs.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            
            # Get logits for masked positions
            mask_logits = logits[mask_index]
            
            t = timesteps[i]
            s = timesteps[i + 1]
            
            # Entropy-based sampling
            if alg == 'entropy':
                confidence, x0 = sample_tokens(
                    mask_logits, 
                    temperature=temperature, 
                    top_p=top_p, 
                    neg_entropy=True
                )
            elif alg == 'maskgit_plus':
                confidence, x0 = sample_tokens(
                    mask_logits, 
                    temperature=temperature, 
                    top_p=top_p
                )
            else:
                confidence, x0 = sample_tokens(
                    mask_logits, 
                    temperature=temperature, 
                    top_p=top_p
                )
            
            # Determine how many tokens to update
            num_mask_token = mask_index.sum().item() / mask_index.shape[0]
            number_transfer_tokens = int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
            
            # Select positions to update based on confidence
            full_confidence = torch.full_like(x, -torch.inf, dtype=logits.dtype, device=device)
            full_confidence[mask_index] = confidence
            
            if number_transfer_tokens > 0:
                if alg_temp is None or alg_temp == 0:
                    # Deterministic: select top-k
                    _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
                else:
                    # Stochastic: sample based on confidence
                    # Only apply softmax to valid (non -inf) positions
                    valid_mask = torch.isfinite(full_confidence)
                    conf_temp = full_confidence.clone()
                    conf_temp[~valid_mask] = -1e10  # Replace -inf with large negative number
                    conf_temp = conf_temp / alg_temp
                    probs = F.softmax(conf_temp, dim=-1)
                    probs[~valid_mask] = 0.0  # Zero out invalid positions
                    
                    # Ensure probabilities sum to > 0
                    prob_sum = probs.sum(dim=-1, keepdim=True)
                    if (prob_sum <= 0).any():
                        # Fallback to uniform distribution over valid positions
                        probs = valid_mask.float()
                        probs = probs / probs.sum(dim=-1, keepdim=True)
                    
                    transfer_index = torch.multinomial(probs, num_samples=number_transfer_tokens)
                
                # Update selected positions
                x_ = torch.zeros_like(x, device=device, dtype=torch.long) + mask_token_id
                x_[mask_index] = x0.clone()
                row_indices = torch.arange(x.size(0), device=device).unsqueeze(1).expand_as(transfer_index)
                x[row_indices, transfer_index] = x_[row_indices, transfer_index]
            
            # Record state every step_interval steps
            if (i + 1) % step_interval == 0 or i == steps - 1:
                generation_history.append(x.clone())
    
    # Decode final output
    generated_ids = x[0, prompt_length:].tolist()
    final_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generation_history, final_output, mask_token_id


def compute_attribution_for_sample(
    model,
    tokenizer,
    sample_text: str,
    category: str,
    max_new_tokens: int = 256,
    steps: int = 32,
    step_interval: int = 4,  # 每隔4步进行一次归因
    temperature: float = 0.8,
    top_p: float = 0.9,
    alg: str = "entropy",
    alg_temp: float = 1.5,
    n_attribution_steps: int = 10,
    device: str = "cuda"
):
    """
    Compute head attribution for a single sample using step-wise attribution
    
    Args:
        step_interval: Compute attribution every N diffusion steps (saves computation)
    
    Returns:
        attribution_scores: shape (n_layers, n_heads)
    """
    print(f"\n{'='*80}")
    print(f"Category: {category}")
    print(f"Prompt: {sample_text[:100]}...")
    print(f"Steps: {steps}, Step Interval: {step_interval}")
    print(f"{'='*80}\n")
    
    # Run generation and get history
    print("Running generation...")
    generation_history, final_output, mask_token_id = run_generation_with_history(
        model, tokenizer, sample_text,
        max_new_tokens=max_new_tokens,
        steps=steps,
        temperature=temperature,
        top_p=top_p,
        alg=alg,
        alg_temp=alg_temp,
        device=device,
        step_interval=step_interval
    )
    
    print(f"Generated output: {final_output[:200]}...")
    print(f"Generation history length: {len(generation_history)} states")
    
    # Initialize attribution computer
    attributor = StepwiseIntegratedGradientsAttributionDream(
        model=model,
        n_steps=n_attribution_steps
    )
    
    n_layers = len(model.layers) if hasattr(model, 'layers') else model.model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads if hasattr(model, 'config') else model.model.config.num_attention_heads
    
    print(f"Computing attribution for {n_layers} layers, {n_heads} heads...")
    
    # Compute attribution across generation history
    importance_scores = attributor.compute_full_generation_attribution(
        generation_history=generation_history,
        mask_token_id=mask_token_id,
        attention_mask="full",
        position_ids=None
    )
    
    # Convert to numpy array: (n_layers, n_heads)
    layer_head_scores = np.zeros((n_layers, n_heads))
    for layer_idx, scores in importance_scores.items():
        layer_head_scores[layer_idx] = scores.cpu().numpy()
    
    return layer_head_scores


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/data/qh_models/Dream-v0-Instruct-7B',
                       help='Path to Dream model')
    parser.add_argument('--samples_per_category', type=int, default=5,
                       help='Number of samples per category')
    parser.add_argument('--max_new_tokens', type=int, default=256,
                       help='Maximum new tokens to generate')
    parser.add_argument('--steps', type=int, default=32,
                       help='Number of diffusion steps')
    parser.add_argument('--step_interval', type=int, default=4,
                       help='Compute attribution every N steps (1=every step, 4=every 4 steps)')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9,
                       help='Top-p sampling')
    parser.add_argument('--alg', type=str, default='entropy',
                       choices=['entropy', 'maskgit_plus', 'topk_margin'],
                       help='Sampling algorithm')
    parser.add_argument('--alg_temp', type=float, default=1.5,
                       help='Algorithm temperature')
    parser.add_argument('--n_attribution_steps', type=int, default=10,
                       help='Number of steps for Integrated Gradients')
    parser.add_argument('--output_dir', type=str, default='./attribution_results_dream',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU ID to use')
    args = parser.parse_args()
    
    # Set GPU
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        print(f"Using GPU: {args.gpu}")
    
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*80)
    print("Step-wise Head Attribution for Dream on Llama-Nemotron Dataset")
    print("="*80)
    print(f"Model: {args.model_path}")
    print(f"Samples per category: {args.samples_per_category}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Diffusion steps: {args.steps}")
    print(f"Step interval: {args.step_interval} (attribution every {args.step_interval} steps)")
    print(f"Temperature: {args.temperature}")
    print(f"Top-p: {args.top_p}")
    print(f"Algorithm: {args.alg} (temp={args.alg_temp})")
    print(f"Attribution steps: {args.n_attribution_steps}")
    print("="*80)
    
    # Load model
    print("\nLoading Dream model...")
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    model = model.to(args.device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    print(f"Model loaded: {model.config.num_hidden_layers} layers, {model.config.num_attention_heads} heads")
    
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
            prompt = prepare_chat_prompt(sample, tokenizer)
            
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
                    max_new_tokens=args.max_new_tokens,
                    steps=args.steps,
                    step_interval=args.step_interval,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    alg=args.alg,
                    alg_temp=args.alg_temp,
                    n_attribution_steps=args.n_attribution_steps,
                    device=args.device
                )
                
                category_attributions.append(attribution_scores)
                
                print(f"  Attribution shape: {attribution_scores.shape}")
                print(f"  Mean score: {attribution_scores.mean():.4f}")
                print(f"  Std score: {attribution_scores.std():.4f}")
                
                # Save individual sample result
                sample_output_file = os.path.join(
                    args.output_dir, 
                    f'attribution_{category}_sample{sample_idx}.npy'
                )
                np.save(sample_output_file, attribution_scores)
                
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
            output_file = os.path.join(args.output_dir, f'attribution_{category}_avg.npy')
            np.save(output_file, avg_attribution)
            print(f"\nSaved {category} attribution to {output_file}")
            
            # Print top heads for this category
            print(f"\nTop 10 most important heads for {category}:")
            flat_scores = avg_attribution.flatten()
            top_indices = np.argsort(flat_scores)[::-1][:10]
            for rank, idx in enumerate(top_indices, 1):
                layer = idx // avg_attribution.shape[1]
                head = idx % avg_attribution.shape[1]
                score = flat_scores[idx]
                print(f"  {rank}. Layer {layer}, Head {head}: {score:.4f}")
    
    # Save summary
    summary_file = os.path.join(args.output_dir, 'attribution_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'categories': list(all_results.keys()),
            'model': args.model_path,
            'config': vars(args),
            'method': 'stepwise_integrated_gradients'
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Attribution computation complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Categories processed: {list(all_results.keys())}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

