"""
Adaptive Sparse Attention Generation for LLaDA

This script demonstrates adaptive sparse attention where different attention heads/groups
have different sparsity levels based on their importance scores.

For testing, head importance is randomly generated.
"""

import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer
import json
import argparse
import time
import sys
import os


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


@torch.no_grad()
def generate(model, prompt, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, SparseD_param=None):
    """
    Generate text using adaptive sparse attention.
    
    Args:
        model: Adaptive mask predictor
        prompt: A tensor of shape (1, L)
        steps: Sampling steps, less than or equal to gen_length
        gen_length: Generated answer length
        block_length: Block length, less than or equal to gen_length
        temperature: Categorical distribution sampling temperature
        cfg_scale: Unsupervised classifier-free guidance scale
        remasking: Remasking strategy ('low_confidence' or 'random')
        mask_id: The token id of [MASK] is 126336
        SparseD_param: Sparse attention parameters including adaptive config
    """
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks
    
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                if SparseD_param is not None:
                    SparseD_param["now_step"] = i + num_block * steps
                    logits = model(x, SparseD_param=SparseD_param).logits
                else:
                    logits = model(x).logits

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

            if remasking == 'low_confidence':
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)  # b, l
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def set_random_seed(seed):
    """Set the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # Add project directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    
    parser = argparse.ArgumentParser(description='Adaptive Sparse Attention Generation for LLaDA')
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-1.5",
                       help="Path to pretrained model")
    parser.add_argument("--seq_len", type=int, default=128,
                       help="Generated sequence length")
    parser.add_argument("--steps", type=int, default=128,
                       help="Number of generation steps")
    parser.add_argument("--block_length", type=int, default=32,
                       help="Block length for generation")
    parser.add_argument("--sampling_alg", type=str, default='low_confidence',
                       choices=['low_confidence', 'random'],
                       help="Remasking strategy")
    
    # Adaptive sparsity arguments
    parser.add_argument("--use_adaptive", action="store_true",
                       help="Use adaptive sparse attention")
    parser.add_argument("--min_sparsity", type=float, default=0.1,
                       help="Minimum sparsity level")
    parser.add_argument("--max_sparsity", type=float, default=0.9,
                       help="Maximum sparsity level")
    parser.add_argument("--importance_strategy", type=str, default='uniform',
                       choices=['uniform', 'normal', 'exponential'],
                       help="Strategy for generating random importance scores")
    
    # Standard sparse attention arguments (for comparison)
    parser.add_argument("--skip", type=float, default=0.2,
                       help="Skip ratio for standard sparse attention")
    parser.add_argument("--select", type=float, default=0.3,
                       help="Selection ratio for standard sparse attention")
    parser.add_argument("--block_size", type=int, default=128,
                       help="Block size for sparse attention")
    
    # Other arguments
    parser.add_argument("--prompt", type=str, default="short_context",
                       help="Prompt key from prompts.json")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--verbose", action="store_true",
                       help="Print detailed information")
    
    args = parser.parse_args()
    
    # Set random seed
    set_random_seed(args.seed)
    
    model_path = args.model_path
    
    print("="*70)
    print("ADAPTIVE SPARSE ATTENTION GENERATION")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Adaptive: {args.use_adaptive}")
    
    if args.use_adaptive:
        # Use adaptive sparse attention model
        print("\n[Loading Adaptive Sparse Model]")
        from models.LLaDA.core.adaptive_sparsed_modeling import AdaptiveLLaDAModelLM
        from models.LLaDA.sparse.adaptive_utils import create_adaptive_sparsity_config, print_adaptive_sparsity_summary
        
        # Load base model first
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        
        # Create adaptive sparsity configuration
        n_layers = config.n_layers
        n_heads = config.n_kv_heads  # Use KV heads for GQA
        
        print(f"\nGenerating random head importance scores...")
        print(f"  Layers: {n_layers}")
        print(f"  KV Heads per layer: {n_heads}")
        print(f"  Strategy: {args.importance_strategy}")
        print(f"  Sparsity range: [{args.min_sparsity}, {args.max_sparsity}]")
        print(f"  Normalize strategy: global_percentile (robust to outliers)")
        print(f"  Output mode: relative weights (mean=1.0)")
        
        adaptive_config = create_adaptive_sparsity_config(
            n_layers=n_layers,
            n_heads=n_heads,
            strategy=args.importance_strategy,
            min_sparsity=args.min_sparsity,
            max_sparsity=args.max_sparsity,
            normalize_strategy='global_percentile',
            output_relative_weights=True,
            seed=args.seed
        )
        
        if args.verbose:
            print_adaptive_sparsity_summary(adaptive_config)
        
        # Load model with adaptive config
        model = AdaptiveLLaDAModelLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True,
            adaptive_config=adaptive_config
        )
        
        # Set up sparse attention parameters
        SparseD_param = {
            'skip': args.skip,
            'select': args.select,  # This will be overridden by adaptive per-head sparsity
            'block_size': args.block_size,
            'new_generation': args.seq_len,
            'whole_steps': args.steps,
            'adaptive': True
        }
        
    else:
        # Use standard sparse model
        print("\n[Loading Standard Sparse Model]")
        from models.LLaDA.core.sparsed_modeling import LLaDAModelLM
        
        model = LLaDAModelLM.from_pretrained(
            model_path, 
            torch_dtype=torch.bfloat16, 
            trust_remote_code=True
        )
        
        SparseD_param = {
            'skip': args.skip,
            'select': args.select,
            'block_size': args.block_size,
            'new_generation': args.seq_len,
            'whole_steps': args.steps
        }
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = model.to("cuda").eval()
    
    # Load or create prompt
    if os.path.exists('prompts.json'):
        with open('prompts.json', 'r', encoding='utf-8') as file:
            data = json.load(file)
        question = data["questions"].get(args.prompt, 
                    "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?")
    else:
        question = "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"
    
    print(f"\n[Question]")
    print(f"{question}")
    
    messages = [{"role": "user", "content": question}]
    prompts = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    prompt_ids = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left")
    input_ids = prompt_ids.input_ids.to(device="cuda")
    
    print(f"\n[Generation Parameters]")
    print(f"  Steps: {args.steps}")
    print(f"  Gen length: {args.seq_len}")
    print(f"  Block length: {args.block_length}")
    print(f"  Sampling: {args.sampling_alg}")
    
    print("\n[Generating...]")
    start_time = time.time()
    output = generate(
        model, input_ids,
        steps=args.steps,
        gen_length=args.seq_len,
        block_length=args.block_length,
        temperature=0,
        remasking=args.sampling_alg,
        SparseD_param=SparseD_param
    )
    end_time = time.time()
    
    answer = tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    
    print("\n" + "="*70)
    print("[RESULT]")
    print("="*70)
    print(f"Answer: {answer}")
    print(f"\nGeneration Time: {end_time - start_time:.4f}s")
    print(f"Tokens/sec: {args.seq_len / (end_time - start_time):.2f}")
    print("="*70)


if __name__ == "__main__":
    main()

