"""
Benchmark script to compare different attention methods:
1. Standard (dense) attention
2. Sparse attention (fixed 30% keep ratio)
3. Adaptive sparse attention (average 30% keep ratio)

Compares inference time with batch size > 1 across multiple examples.
"""

import torch
import numpy as np
import time
import argparse
import json
from typing import List, Dict
from transformers import AutoTokenizer
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def add_gumbel_noise(logits, temperature):
    """Gumbel noise for sampling."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """Calculate number of tokens to transfer at each step."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    return num_transfer_tokens


@torch.no_grad()
def generate_batch(model, prompts, steps=128, gen_length=128, block_length=128, 
                   temperature=0., remasking='low_confidence', mask_id=126336,
                   SparseD_param=None):
    """
    Generate text for a batch of prompts.
    
    Args:
        model: The model to use
        prompts: Tensor of shape (batch_size, prompt_len)
        SparseD_param: Sparse attention parameters (None for dense)
    
    Returns:
        output: Generated sequences (tensor)
        prompt_len: Length of the prompt
        generation_time: Time taken for generation
    """
    batch_size = prompts.shape[0]
    device = model.device
    prompt_len = prompts.shape[1]
    
    x = torch.full((batch_size, prompts.shape[1] + gen_length), mask_id, dtype=torch.long).to(device)
    x[:, :prompts.shape[1]] = prompts.clone()
    
    prompt_index = (x != mask_id)
    
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    
    # Start timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompts.shape[1] + num_block * block_length: 
                              prompts.shape[1] + (num_block + 1) * block_length] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            
            if SparseD_param is not None:
                SparseD_param["now_step"] = i + num_block * steps_per_block
                logits = model(x, SparseD_param=SparseD_param).logits
            else:
                logits = model(x).logits
            
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)
            
            if remasking == 'low_confidence':
                import torch.nn.functional as F
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            else:
                raise NotImplementedError(remasking)
            
            x0_p[:, prompts.shape[1] + (num_block + 1) * block_length:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)
            
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]
    
    # End timing
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end_time = time.time()
    generation_time = end_time - start_time
    
    return x, prompt_len, generation_time


def prepare_test_questions(num_samples: int = 10) -> List[str]:
    """Prepare a list of test questions."""
    questions = [
        "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
        "A store has 15 apples. If they sell 7 apples, how many apples are left?",
        "John has 8 pencils. Mary gives him 5 more pencils. How many pencils does John have now?",
        "There are 20 students in a class. If 3 students are absent, how many students are present?",
        "A basket has 12 oranges. If you add 8 more oranges, how many oranges are in the basket?",
        "Sarah reads 6 pages of a book every day. How many pages will she read in 5 days?",
        "A box contains 25 chocolates. If Tom eats 9 chocolates, how many chocolates remain?",
        "Lisa has 14 flowers. She gives 6 flowers to her friend. How many flowers does Lisa have left?",
        "There are 18 birds on a tree. If 7 birds fly away, how many birds remain on the tree?",
        "A farmer has 30 chickens. If he buys 12 more chickens, how many chickens does he have in total?",
    ]
    
    # Repeat questions if we need more samples
    if num_samples > len(questions):
        questions = questions * (num_samples // len(questions) + 1)
    
    return questions[:num_samples]


def benchmark_method(model, tokenizer, questions: List[str], batch_size: int,
                     method_name: str, SparseD_param=None, 
                     seq_len=128, steps=128, block_length=32, print_samples=True):
    """
    Benchmark a specific attention method.
    
    Returns:
        Dict with timing statistics
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking: {method_name}")
    print(f"{'='*70}")
    
    # Prepare batches
    num_questions = len(questions)
    num_batches = (num_questions + batch_size - 1) // batch_size
    
    batch_times = []
    total_tokens = 0
    sample_outputs = []  # Store some sample outputs
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_questions)
        batch_questions = questions[start_idx:end_idx]
        actual_batch_size = len(batch_questions)
        
        # Prepare prompts
        messages_list = [{"role": "user", "content": q} for q in batch_questions]
        prompts_list = [tokenizer.apply_chat_template([m], add_generation_prompt=True, tokenize=False) 
                       for m in messages_list]
        
        # Tokenize with padding
        encoded = tokenizer(prompts_list, return_tensors="pt", padding=True, padding_side="left")
        input_ids = encoded.input_ids.to(model.device)
        
        print(f"  Batch {batch_idx + 1}/{num_batches} (size={actual_batch_size})...", end=" ", flush=True)
        
        # Generate
        try:
            output, prompt_len, gen_time = generate_batch(
                model, input_ids,
                steps=steps,
                gen_length=seq_len,
                block_length=block_length,
                temperature=0,
                remasking='low_confidence',
                SparseD_param=SparseD_param
            )
            
            batch_times.append(gen_time)
            total_tokens += actual_batch_size * seq_len
            
            print(f"Time: {gen_time:.3f}s")
            
            # Store first batch outputs for printing
            if batch_idx == 0 and print_samples:
                for i in range(min(2, actual_batch_size)):  # Print first 2 samples
                    generated_text = tokenizer.decode(output[i, prompt_len:], skip_special_tokens=True)
                    sample_outputs.append({
                        'question': batch_questions[i],
                        'generated': generated_text
                    })
        
        except Exception as e:
            print(f"ERROR: {str(e)}")
            continue
    
    # Print sample outputs
    if print_samples and sample_outputs:
        print(f"\n  {'='*68}")
        print(f"  Sample Outputs (first batch):")
        print(f"  {'='*68}")
        for idx, sample in enumerate(sample_outputs):
            print(f"\n  Sample {idx + 1}:")
            print(f"  Question: {sample['question']}")
            print(f"  Generated: {sample['generated'][:200]}...")  # Print first 200 chars
            print(f"  {'-'*68}")
    
    # Calculate statistics
    if len(batch_times) == 0:
        return None
    
    total_time = sum(batch_times)
    avg_time_per_batch = np.mean(batch_times)
    std_time_per_batch = np.std(batch_times)
    avg_time_per_sample = total_time / num_questions
    tokens_per_sec = total_tokens / total_time
    
    results = {
        'method': method_name,
        'total_time': total_time,
        'avg_time_per_batch': avg_time_per_batch,
        'std_time_per_batch': std_time_per_batch,
        'avg_time_per_sample': avg_time_per_sample,
        'tokens_per_sec': tokens_per_sec,
        'num_batches': len(batch_times),
        'num_samples': num_questions,
        'batch_size': batch_size,
        'batch_times': batch_times
    }
    
    print(f"\n  Results:")
    print(f"    Total time: {total_time:.3f}s")
    print(f"    Avg time/batch: {avg_time_per_batch:.3f}±{std_time_per_batch:.3f}s")
    print(f"    Avg time/sample: {avg_time_per_sample:.3f}s")
    print(f"    Throughput: {tokens_per_sec:.2f} tokens/sec")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Benchmark different attention methods')
    parser.add_argument('--model_path', type=str, default='GSAI-ML/LLaDA-1.5',
                       help='Path to the model')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for inference')
    parser.add_argument('--seq_len', type=int, default=128,
                       help='Generation length')
    parser.add_argument('--steps', type=int, default=128,
                       help='Sampling steps')
    parser.add_argument('--block_length', type=int, default=32,
                       help='Block length')
    parser.add_argument('--block_size', type=int, default=128,
                       help='Block size for sparse attention')
    
    # Test configuration
    parser.add_argument('--test_dense', action='store_true',
                       help='Test dense attention (no sparsity)')
    parser.add_argument('--test_sparse', action='store_true',
                       help='Test standard sparse attention')
    parser.add_argument('--test_adaptive', action='store_true',
                       help='Test adaptive sparse attention')
    parser.add_argument('--test_all', action='store_true',
                       help='Test all methods')
    
    # Sparsity parameters
    parser.add_argument('--sparse_keep_ratio', type=float, default=0.3,
                       help='Keep ratio for standard sparse (0.3 = 30%%)')
    parser.add_argument('--adaptive_avg_keep', type=float, default=0.3,
                       help='Average keep ratio for adaptive sparse')
    
    parser.add_argument('--output', type=str, default='benchmark_results.json',
                       help='Output file for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--print_outputs', action='store_true',
                       help='Print sample outputs for each method')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print("="*70)
    print("ATTENTION METHODS BENCHMARK")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Num samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Steps: {args.steps}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("="*70)
    
    # Prepare test questions
    questions = prepare_test_questions(args.num_samples)
    print(f"\nPrepared {len(questions)} test questions")
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    # Determine which methods to test
    test_methods = []
    if args.test_all:
        test_methods = ['dense', 'sparse', 'adaptive']
    else:
        if args.test_dense:
            test_methods.append('dense')
        if args.test_sparse:
            test_methods.append('sparse')
        if args.test_adaptive:
            test_methods.append('adaptive')
    
    if not test_methods:
        print("\nNo methods selected! Use --test_all or --test_dense/--test_sparse/--test_adaptive")
        return
    
    print(f"\nMethods to test: {', '.join(test_methods)}")
    
    all_results = []
    
    # Test each method
    for method in test_methods:
        print(f"\n{'#'*70}")
        print(f"# Method: {method.upper()}")
        print(f"{'#'*70}")
        
        if method == 'dense':
            # Dense attention (no sparsity)
            print("\nLoading model with dense attention...")
            from models.LLaDA.core.sparsed_modeling import LLaDAModelLM
            model = LLaDAModelLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
            
            results = benchmark_method(
                model, tokenizer, questions, args.batch_size,
                method_name="Dense Attention",
                SparseD_param=None,
                seq_len=args.seq_len,
                steps=args.steps,
                block_length=args.block_length,
                print_samples=args.print_outputs
            )
            
        elif method == 'sparse':
            # Standard sparse attention
            print("\nLoading model with standard sparse attention...")
            from models.LLaDA.core.sparsed_modeling import LLaDAModelLM
            model = LLaDAModelLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            )
            model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
            
            # Calculate skip and select to achieve target keep ratio
            # keep_ratio = select, so select = keep_ratio
            select_ratio = args.sparse_keep_ratio
            
            SparseD_param = {
                'skip': 0.2,
                'select': select_ratio,
                'block_size': args.block_size,
                'new_generation': args.seq_len,
                'whole_steps': args.steps
            }
            
            results = benchmark_method(
                model, tokenizer, questions, args.batch_size,
                method_name=f"Standard Sparse (keep={args.sparse_keep_ratio:.0%})",
                SparseD_param=SparseD_param,
                seq_len=args.seq_len,
                steps=args.steps,
                block_length=args.block_length,
                print_samples=args.print_outputs
            )
            
        elif method == 'adaptive':
            # Adaptive sparse attention
            print("\nLoading model with adaptive sparse attention...")
            from models.LLaDA.core.adaptive_sparsed_modeling import AdaptiveLLaDAModelLM
            from models.LLaDA.sparse.adaptive_utils import create_adaptive_sparsity_config
            from transformers import AutoConfig
            
            config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
            n_layers = config.n_layers
            n_heads = config.n_kv_heads
            
            # Create adaptive config with target average keep ratio
            # min and max are set so average ≈ target
            target_keep = args.adaptive_avg_keep
            min_keep = max(0.1, target_keep - 0.2)
            max_keep = min(0.9, target_keep + 0.2)
            adaptive_config = create_adaptive_sparsity_config(
                n_layers=n_layers,
                n_heads=n_heads,
                strategy='uniform',
                base_sparsity=1.0 - target_keep,  # sparsity = 1 - keep_ratio
                min_sparsity=1.0 - max_keep,
                max_sparsity=1.0 - min_keep,
                seed=args.seed
            )
            
            # Print average keep ratio
            avg_keep = sum(layer.mean().item() for layer in adaptive_config['sparsity_levels'].values()) / n_layers
            print(f"  Average keep ratio: {avg_keep:.2%}")
            
            model = AdaptiveLLaDAModelLM.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                adaptive_config=adaptive_config
            )
            model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
            
            SparseD_param = {
                'skip': 0.2,
                'select': args.adaptive_avg_keep,
                'block_size': args.block_size,
                'new_generation': args.seq_len,
                'whole_steps': args.steps,
                'adaptive': True
            }
            
            results = benchmark_method(
                model, tokenizer, questions, args.batch_size,
                method_name=f"Adaptive Sparse (avg keep={avg_keep:.0%})",
                SparseD_param=SparseD_param,
                seq_len=args.seq_len,
                steps=args.steps,
                block_length=args.block_length,
                print_samples=args.print_outputs
            )
        
        if results:
            all_results.append(results)
        
        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Print comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    if len(all_results) > 1:
        # Use first method as baseline
        baseline = all_results[0]
        
        print(f"\n{'Method':<40} {'Time/sample':<15} {'Speedup':<12} {'Throughput'}")
        print("-"*70)
        
        for result in all_results:
            speedup = baseline['avg_time_per_sample'] / result['avg_time_per_sample']
            print(f"{result['method']:<40} {result['avg_time_per_sample']:>6.3f}s         "
                  f"{speedup:>6.2f}x       {result['tokens_per_sec']:>6.1f} tok/s")
    
    # Save results
    output_data = {
        'config': {
            'model_path': args.model_path,
            'num_samples': args.num_samples,
            'batch_size': args.batch_size,
            'seq_len': args.seq_len,
            'steps': args.steps,
            'seed': args.seed,
        },
        'results': all_results
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ Results saved to {args.output}")
    print("="*70)


if __name__ == "__main__":
    main()

