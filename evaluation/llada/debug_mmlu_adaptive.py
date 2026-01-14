#!/usr/bin/env python3
"""
Debug script to diagnose why adaptive mode works on GSM8K but not MMLU.

This script compares:
1. Sparse vs Adaptive attention patterns
2. Effect of mc_num (sampling times) on stability
3. Impact of recompute_mask_each_call on likelihood scores
"""

import torch
import sys
sys.path.insert(0, '/home/qiheng/Projects/adaptive-dllm')

from models.LLaDA.core.sparsed_modeling import LLaDAModelLM as SparseLLaDAModelLM
from models.LLaDA.core.adaptive_sparsed_modeling import AdaptiveLLaDAModelLM
from models.LLaDA.sparse.adaptive_utils import create_adaptive_sparsity_config
from transformers import AutoTokenizer, AutoConfig
import torch.nn.functional as F


def load_models(model_path, importance_path):
    """Load sparse and adaptive models"""
    print(f"Loading models from: {model_path}")
    
    # Load sparse model
    sparse_model = SparseLLaDAModelLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map='cuda'
    )
    sparse_model.eval()
    
    # Load adaptive model
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    n_layers = config.n_layers
    n_heads = config.n_heads
    
    importance_data = torch.load(importance_path, weights_only=False)
    importance_scores = importance_data['importance_scores']
    
    adaptive_config = create_adaptive_sparsity_config(
        n_layers=n_layers,
        n_heads=n_heads,
        importance_scores=importance_scores,
        min_sparsity=0.15,
        max_sparsity=0.85,
        normalize_strategy='global_percentile',
        output_relative_weights=True,
        seed=42
    )
    
    adaptive_model = AdaptiveLLaDAModelLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        adaptive_config=adaptive_config,
        device_map='cuda'
    )
    adaptive_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    return sparse_model, adaptive_model, tokenizer


def test_likelihood_stability(model, tokenizer, prefix_text, target_text, model_type, mc_num_list=[1, 5, 10]):
    """Test likelihood estimation stability with different mc_num"""
    print(f"\n{'='*70}")
    print(f"Testing {model_type} model")
    print(f"{'='*70}")
    print(f"Prefix: {prefix_text[:100]}...")
    print(f"Target: {target_text}")
    
    prefix = torch.tensor(tokenizer(prefix_text)["input_ids"], device='cuda')
    target = torch.tensor(tokenizer(target_text)["input_ids"], device='cuda')
    
    results = {}
    
    for mc_num in mc_num_list:
        print(f"\n  mc_num={mc_num}:")
        likelihoods = []
        
        # Run multiple times to check variance
        for run in range(3):
            ll = compute_loglikelihood(
                model, tokenizer, prefix, target,
                mc_num=mc_num,
                batch_size=1,
                model_type=model_type
            )
            likelihoods.append(ll)
            print(f"    Run {run+1}: {ll:.4f}")
        
        mean_ll = sum(likelihoods) / len(likelihoods)
        std_ll = (sum((x - mean_ll)**2 for x in likelihoods) / len(likelihoods)) ** 0.5
        print(f"    Mean: {mean_ll:.4f}, Std: {std_ll:.4f}")
        results[mc_num] = {'mean': mean_ll, 'std': std_ll, 'values': likelihoods}
    
    return results


def compute_loglikelihood(model, tokenizer, prefix, target, mc_num=1, batch_size=1, model_type='sparse'):
    """Compute log-likelihood of target given prefix"""
    seq = torch.cat([prefix, target]).unsqueeze(0)  # (1, L)
    seq = seq.repeat(batch_size, 1)
    prompt_index = torch.arange(seq.shape[1], device=seq.device) < len(prefix)
    
    # Sparse parameters
    sparse_param = {
        'skip': 0.2,
        'select': 0.3,
        'block_size': 32,
        'new_generation': 256,
        'whole_steps': 256,
        'now_step': 256,  # Fixed for likelihood
        'recompute_mask_each_call': True
    }
    if model_type == 'adaptive':
        sparse_param['adaptive'] = True
    
    mask_id = 126336
    loss_acc = []
    
    for _ in range(mc_num // batch_size):
        # Perturb sequence
        perturbed_seq, p_mask = forward_process(seq, prompt_index, mask_id)
        mask_indices = perturbed_seq == mask_id
        
        # Get logits
        with torch.no_grad():
            if model_type in ['sparse', 'adaptive']:
                logits = model(perturbed_seq, SparseD_param=sparse_param).logits
            else:
                logits = model(perturbed_seq).logits
        
        # Compute loss
        loss = F.cross_entropy(
            logits[mask_indices],
            seq[mask_indices],
            reduction='none'
        ) / p_mask[mask_indices]
        loss = loss.sum() / batch_size
        loss_acc.append(loss.item())
    
    return -sum(loss_acc) / len(loss_acc)


def forward_process(batch, prompt_index, mask_id):
    """Same as eval_llada.py _forward_process"""
    b, l = batch.shape
    target_len = int((~prompt_index).sum().item())
    k = torch.randint(1, target_len + 1, (), device=batch.device)
    
    x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
    x = ((x - 1) % target_len) + 1
    
    indices = torch.arange(target_len, device=batch.device).unsqueeze(0).repeat(b, 1)
    is_mask = indices < x.unsqueeze(1)
    
    for i in range(b):
        is_mask[i] = is_mask[i][torch.randperm(target_len, device=batch.device)]
    
    prefix_mask = torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device)
    is_mask = torch.cat([prefix_mask, is_mask], dim=1)
    noisy_batch = torch.where(is_mask, mask_id, batch)
    
    return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)


def main():
    model_path = "/data/qh_models/LLaDA-1.5"
    importance_path = "/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada-1_5_loss_gateIG_neg/head_importance.pt"
    
    print(f"Loading models...")
    sparse_model, adaptive_model, tokenizer = load_models(model_path, importance_path)
    
    # Test MMLU-style example
    mmlu_prefix = "Question: What is the capital of France?\nA. London\nB. Berlin\nC. Paris\nD. Rome\nAnswer:"
    mmlu_target_correct = " C"
    mmlu_target_wrong = " A"
    
    print("\n" + "="*80)
    print("MMLU-STYLE TEST: Correct answer")
    print("="*80)
    
    sparse_results = test_likelihood_stability(
        sparse_model, tokenizer, mmlu_prefix, mmlu_target_correct, 
        model_type='sparse', mc_num_list=[1, 5, 10]
    )
    
    adaptive_results = test_likelihood_stability(
        adaptive_model, tokenizer, mmlu_prefix, mmlu_target_correct,
        model_type='adaptive', mc_num_list=[1, 5, 10]
    )
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nCorrect answer ('{mmlu_target_correct}'):")
    print(f"  Sparse  (mc_num=1):  {sparse_results[1]['mean']:.4f} ± {sparse_results[1]['std']:.4f}")
    print(f"  Adaptive (mc_num=1): {adaptive_results[1]['mean']:.4f} ± {adaptive_results[1]['std']:.4f}")
    print(f"  Difference: {adaptive_results[1]['mean'] - sparse_results[1]['mean']:.4f}")
    
    print(f"\n  Sparse  (mc_num=10):  {sparse_results[10]['mean']:.4f} ± {sparse_results[10]['std']:.4f}")
    print(f"  Adaptive (mc_num=10): {adaptive_results[10]['mean']:.4f} ± {adaptive_results[10]['std']:.4f}")
    print(f"  Difference: {adaptive_results[10]['mean'] - sparse_results[10]['mean']:.4f}")
    
    # Analysis
    print(f"\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    if sparse_results[1]['std'] > 0.5 or adaptive_results[1]['std'] > 0.5:
        print("⚠️  WARNING: High variance with mc_num=1!")
        print("   This suggests mc_num=1 is too low for stable likelihood estimation.")
    
    if adaptive_results[1]['mean'] < sparse_results[1]['mean']:
        print("⚠️  WARNING: Adaptive likelihood < Sparse likelihood!")
        print("   Possible causes:")
        print("   1. Negated importance scores are incorrectly interpreted")
        print("   2. Adaptive masks are too aggressive for likelihood tasks")
        print("   3. recompute_mask_each_call causes instability with perturbed sequences")
    
    print("\n💡 RECOMMENDATIONS:")
    print("   1. Try increasing mc_num from 1 to 10+ for MMLU")
    print("   2. Compare with IMPORTANCE_TAG='loss_gateIG' (non-negated)")
    print("   3. Check if adaptive works better with recompute_mask_each_call=False")


if __name__ == '__main__':
    main()

