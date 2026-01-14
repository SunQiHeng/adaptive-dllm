#!/usr/bin/env python3
"""
LLaDA Evaluation with lm-eval
Supports: standard, sparse, and adaptive sparse attention
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import accelerate
import torch
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoTokenizer

# Import generation functions from models/LLaDA/generation/
from models.LLaDA.generation.generate import generate as standard_generate
from models.LLaDA.generation.sparsed_generate import generate as sparse_generate
from models.LLaDA.generation.adaptive_sparsed_generate import generate as adaptive_generate


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_adaptive_config_stats(adaptive_config, select, n_layers, n_heads, model_name="Model"):
    """
    Print statistics about adaptive sparsity configuration.
    
    Args:
        adaptive_config: The adaptive configuration dict
        select: The select parameter (target average keep ratio)
        n_layers: Number of layers
        n_heads: Number of KV heads per layer
        model_name: Name of the model for display
    """
    print("\n" + "=" * 80)
    print(f"📊 {model_name} Adaptive Sparsity Configuration Statistics")
    print("=" * 80)
    
    sparsity_levels = adaptive_config['sparsity_levels']
    metadata = adaptive_config.get('metadata', {})
    
    # Check if we're using relative weights or absolute keep_ratios
    output_relative_weights = metadata.get('output_relative_weights', True)
    
    if output_relative_weights:
        print(f"Mode: Relative Weights (mean=1.0, multiply by select at inference)")
    else:
        print(f"Mode: Absolute Keep Ratios (pre-computed)")
    
    print(f"Target select: {select:.3f} ({select*100:.1f}%)")
    print(f"Layers: {n_layers}, KV Heads per layer: {n_heads}")
    print(f"Total heads: {n_layers * n_heads}")
    
    # Collect all weights/keep_ratios
    all_values = []
    layer_means = []
    
    print("\n" + "-" * 80)
    print("Per-Layer Statistics:")
    print("-" * 80)
    
    for layer_idx in range(n_layers):
        values = sparsity_levels[layer_idx]
        print(values)

        # `values` is either:
        # - relative weights (mean≈1.0) if output_relative_weights=True
        # - absolute keep-ratios (mean≈select) if output_relative_weights=False
        weights_tensor_layer = values.to(torch.float32)

        if output_relative_weights:
            keep_tensor_layer = torch.clamp(weights_tensor_layer * float(select), max=1.0)
            weight_mean = weights_tensor_layer.mean().item()
            weight_min = weights_tensor_layer.min().item()
            weight_max = weights_tensor_layer.max().item()

            keep_mean_layer = keep_tensor_layer.mean().item()
            keep_min_layer = keep_tensor_layer.min().item()
            keep_max_layer = keep_tensor_layer.max().item()

            all_values.append(values)  # keep global weight stats correct
            layer_means.append(keep_mean_layer)  # store *keep-ratio* means for layer-wise variation

            print(
                f"Layer {layer_idx:2d}: weight_mean={weight_mean:.4f} (range=[{weight_min:.3f}, {weight_max:.3f}]) "
                f"→ keep_ratio_mean={keep_mean_layer:.4f} ({keep_mean_layer*100:.1f}%) "
                f"(range=[{keep_min_layer:.3f}, {keep_max_layer:.3f}])"
            )
        else:
            # Absolute keep ratios
            keep_mean_layer = weights_tensor_layer.mean().item()
            keep_min_layer = weights_tensor_layer.min().item()
            keep_max_layer = weights_tensor_layer.max().item()

            all_values.append(values)
            layer_means.append(keep_mean_layer)

            print(
                f"Layer {layer_idx:2d}: keep_ratio_mean={keep_mean_layer:.4f} ({keep_mean_layer*100:.1f}%), "
                f"range=[{keep_min_layer:.3f}, {keep_max_layer:.3f}]"
            )
    
    # Global statistics
    #
    # NOTE:
    # - In relative-weights mode, `all_values` are the *weights* (mean should be ≈1.0).
    # - Keep-ratios are computed at inference by: keep_ratio = clamp(weights * select, max=1.0).
    #
    # Previously this function overwrote `all_values` with keep-ratios and then printed them as
    # "Relative weights", which was misleading (e.g., showing mean≈select).
    weights_tensor = torch.cat(all_values).to(torch.float32)
    global_mean = weights_tensor.mean().item()
    global_min = weights_tensor.min().item()
    global_max = weights_tensor.max().item()
    global_std = weights_tensor.std().item()

    keep_tensor = torch.clamp(weights_tensor * float(select), max=1.0)
    keep_mean = keep_tensor.mean().item()
    keep_min = keep_tensor.min().item()
    keep_max = keep_tensor.max().item()
    
    print("\n" + "-" * 80)
    print("Global Statistics:")
    print("-" * 80)
    
    if output_relative_weights:
        print(f"Relative weights:")
        print(f"  Mean:   {global_mean:.4f} (should be ≈1.0)")
        print(f"  Std:    {global_std:.4f}")
        print(f"  Range:  [{global_min:.3f}, {global_max:.3f}]")
        print(f"\nActual keep_ratios (weights × select={select}):")
        print(f"  Mean:   {keep_mean:.4f} ({keep_mean*100:.1f}%)")
        print(f"  Target: {select:.4f} ({select*100:.1f}%)")
        print(f"  Deviation: {abs(keep_mean - select):.4f} ({abs(keep_mean - select)*100:.2f}%)")
        print(f"  Range:  [{keep_min:.3f}, {keep_max:.3f}]")
        
        # Count heads that will hit upper limit (keep_ratio > 1.0 after scaling)
        clamped_count = (weights_tensor * float(select) > 1.0).sum().item()
        total_heads = n_layers * n_heads
        print(f"  Heads hitting upper limit (>1.0): {clamped_count}/{total_heads} ({clamped_count/total_heads*100:.1f}%)")
        
        if abs(keep_mean - select) < 0.01:
            print(f"  ✅ Mean matches target (deviation < 1%)")
        elif abs(keep_mean - select) < 0.05:
            print(f"  ⚠️  Mean has slight deviation (1-5%)")
        else:
            print(f"  ❌ Mean deviates significantly (>5%)")
    else:
        print(f"Keep ratios:")
        print(f"  Mean:   {global_mean:.4f} ({global_mean*100:.1f}%)")
        print(f"  Std:    {global_std:.4f}")
        print(f"  Range:  [{global_min:.3f}, {global_max:.3f}]")
    
    # Layer-wise variation
    layer_means_tensor = torch.tensor(layer_means)
    layer_mean_std = layer_means_tensor.std().item()
    layer_mean_min = layer_means_tensor.min().item()
    layer_mean_max = layer_means_tensor.max().item()
    
    print("\n" + "-" * 80)
    print("Layer-wise Variation:")
    print("-" * 80)
    
    if output_relative_weights:
        # `layer_means` already stores per-layer *keep-ratio* means (not weights)
        actual_layer_min = layer_mean_min
        actual_layer_max = layer_mean_max
        print(f"Layer means range: [{layer_mean_min:.4f}, {layer_mean_max:.4f}]")
        print(f"Layer means std:   {layer_mean_std:.4f}")
        print(f"Actual keep_ratio range across layers: [{actual_layer_min:.3f}, {actual_layer_max:.3f}]")
        print(f"Variation: {(actual_layer_max - actual_layer_min)*100:.1f}% spread")
    else:
        print(f"Layer means range: [{layer_mean_min:.4f}, {layer_mean_max:.4f}]")
        print(f"Layer means std:   {layer_mean_std:.4f}")
        print(f"Variation: {(layer_mean_max - layer_mean_min)*100:.1f}% spread")
    
    print("=" * 80 + "\n")


@register_model("llada_eval")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='GSAI-ML/LLaDA-8B-Base',
        model_type='standard',  # 'standard', 'sparse', 'adaptive'
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=False,
        cfg=0.,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        # Sparse params
        skip=0.2,
        select=0.3,
        block_size=128,
        # Adaptive params
        adaptive_config_path=None,
        importance_source='precomputed',  # 'precomputed', 'uniform', 'normal', or custom path
        precomputed_importance_path=None,
        min_sparsity=0.15,  # Updated: optimized for global_percentile normalization
        max_sparsity=0.85,  # Updated: optimized for global_percentile normalization
        # Likelihood eval params (for multiple-choice tasks like MMLU)
        #
        # NOTE:
        # - Multiple-choice tasks like MMLU go through `loglikelihood()` (not `generate_until()`).
        # - The sparse/adaptive attention implementations only start applying sparse attention
        #   after a warmup window controlled by (now_step, whole_steps, skip).
        # - Historically `get_logits()` forced now_step=0 for likelihood scoring, which keeps the
        #   model in warmup -> sparse/adaptive behave like standard attention.
        #
        # To make sparse/adaptive affect likelihood scoring, set:
        # - likelihood_now_step: e.g. =steps (or any value > end_time)
        # - recompute_mask_each_call: True (masks depend on content; caching across examples is wrong)
        likelihood_now_step=None,
        recompute_mask_each_call=False,
        device="cuda",
        **kwargs,
    ):
        super().__init__()
        
        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})
        
        # Load model based on type
        if model_type == 'standard':
            from models.LLaDA.core.modeling import LLaDAModelLM
            self.model = LLaDAModelLM.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True,
                **model_kwargs
            )
            self.sparse_param = None
            
        elif model_type == 'sparse':
            from models.LLaDA.core.sparsed_modeling import LLaDAModelLM
            self.model = LLaDAModelLM.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True,
                **model_kwargs
            )
            self.sparse_param = {
                'skip': skip,
                'select': select,
                'block_size': block_size,
                'new_generation': gen_length,
                'whole_steps': steps
            }
            
        elif model_type == 'adaptive':
            from models.LLaDA.core.adaptive_sparsed_modeling import AdaptiveLLaDAModelLM
            from models.LLaDA.sparse.adaptive_utils import create_adaptive_sparsity_config
            from transformers import AutoConfig
            import json
            
            # Determine importance scores source
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            n_layers = config.n_layers
            # Use query heads here because attribution is typically produced per attention (query) head.
            # Adaptive blocks can accept per-query-head sparsity and will aggregate to KV heads for GQA.
            n_heads = config.n_heads
            
            importance_scores = None
            
            # Option 1: Load from explicit config file path
            if adaptive_config_path and os.path.exists(adaptive_config_path):
                if adaptive_config_path.endswith('.json'):
                    with open(adaptive_config_path, 'r') as f:
                        adaptive_config = json.load(f)
                    if 'keep_mask' in adaptive_config:
                        adaptive_config['keep_mask'] = torch.tensor(adaptive_config['keep_mask'], dtype=torch.bool)
                else:
                    adaptive_config = torch.load(adaptive_config_path, weights_only=False)
                print(f"✓ Loaded adaptive config from: {adaptive_config_path}")
            
            # Option 2: Load from importance source specification
            elif importance_source == 'precomputed':
                # Use pre-computed importance scores from attribution
                llada_importance_path = precomputed_importance_path or '/home/qiheng/Projects/adaptive-dllm/configs/head_importance_llada_base_margin/head_importance.pt'
                if os.path.exists(llada_importance_path):
                    print(f"✓ Loading pre-computed importance scores from: {llada_importance_path}")
                    importance_data = torch.load(llada_importance_path, weights_only=False)
                    importance_scores = importance_data['importance_scores']
                else:
                    raise FileNotFoundError(f"Pre-computed importance file not found: {llada_importance_path}")
                
                adaptive_config = create_adaptive_sparsity_config(
                    n_layers=n_layers,
                    n_heads=n_heads,
                    importance_scores=importance_scores,  # Use precomputed
                    min_sparsity=min_sparsity,
                    max_sparsity=max_sparsity,
                    normalize_strategy='global_percentile',
                    output_relative_weights=True,
                    seed=42
                )
                print(f"✓ Created adaptive config using PRE-COMPUTED importance scores")
                
                # Print detailed statistics
                print_adaptive_config_stats(adaptive_config, select, n_layers, n_heads, "LLaDA")
            
            elif importance_source == "all_ones":
                # Fair sanity: all heads have equal relative weight (=1.0), so keep_ratio_per_head == select.
                # This should closely match sparse mode when skip/select/block_size are the same.
                print("✓ Using ALL-ONES importance (equal head weights = 1.0) for fair sanity check")
                adaptive_config = {
                    "importance_scores": {i: torch.ones(n_heads) for i in range(n_layers)},
                    "sparsity_levels": {i: torch.ones(n_heads) for i in range(n_layers)},  # relative weights (mean=1.0)
                    "metadata": {
                        "strategy": "all_ones",
                        "normalize_strategy": None,
                        "output_relative_weights": True,
                        "note": "All heads have equal relative weight (=1.0). keep_ratio = select after clamping.",
                    },
                }
                print("✓ Created adaptive config using ALL-ONES importance")
                print_adaptive_config_stats(adaptive_config, select, n_layers, n_heads, "LLaDA")

            elif importance_source in ['uniform', 'normal', 'random']:
                # Generate random importance with specified distribution
                print(f"✓ Generating RANDOM importance scores with '{importance_source}' distribution")
                adaptive_config = create_adaptive_sparsity_config(
                    n_layers=n_layers,
                    n_heads=n_heads,
                    importance_scores=None,  # Generate random
                    strategy=importance_source,
                    min_sparsity=min_sparsity,
                    max_sparsity=max_sparsity,
                    normalize_strategy='global_percentile',
                    output_relative_weights=True,
                    seed=42
                )
                print(f"✓ Created adaptive config using RANDOM '{importance_source}' importance")
                
                # Print detailed statistics
                print_adaptive_config_stats(adaptive_config, select, n_layers, n_heads, "LLaDA")
            
            elif os.path.exists(importance_source):
                # Custom importance file path
                print(f"✓ Loading custom importance scores from: {importance_source}")
                importance_data = torch.load(importance_source, weights_only=False)
                importance_scores = importance_data['importance_scores']
                
                adaptive_config = create_adaptive_sparsity_config(
                    n_layers=n_layers,
                    n_heads=n_heads,
                    importance_scores=importance_scores,
                    min_sparsity=min_sparsity,
                    max_sparsity=max_sparsity,
                    normalize_strategy='global_percentile',
                    output_relative_weights=True,
                    seed=42
                )
                print(f"✓ Created adaptive config using CUSTOM importance from: {importance_source}")
                
                # Print detailed statistics
                print_adaptive_config_stats(adaptive_config, select, n_layers, n_heads, "LLaDA")
            
            else:
                raise ValueError(
                    f"Invalid importance_source: {importance_source}. "
                    f"Must be 'precomputed', 'all_ones', 'uniform', 'normal', 'random', or a valid file path."
                )
            
            self.model = AdaptiveLLaDAModelLM.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True,
                adaptive_config=adaptive_config,
                **model_kwargs
            )
            
            self.sparse_param = {
                'skip': skip,
                'select': select,
                'block_size': block_size,
                'new_generation': gen_length,
                'whole_steps': steps,
                'adaptive': True
            }
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        self.model.eval()
        self.model_type = model_type
        
        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)
            self._rank = 0
            self._world_size = 1
        
        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy
        
        self.cfg = cfg
        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        self.likelihood_now_step = likelihood_now_step
        self.recompute_mask_each_call = recompute_mask_each_call
        
        print(f"\n{'='*70}")
        print(f"LLaDA Evaluation Setup")
        print(f"{'='*70}")
        print(f"Model type: {model_type}")
        print(f"Model path: {model_path}")
        print(f"Steps: {steps}, Gen length: {gen_length}, Block length: {block_length}")
        if model_type in ['sparse', 'adaptive']:
            print(f"Sparse params: skip={skip}, select={select}, block_size={block_size}")
        print(f"{'='*70}\n")
    
    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size
    
    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape
        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)
        
        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        
        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)
        
        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]
        
        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)
        noisy_batch = torch.where(is_mask, self.mask_id, batch)
        
        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)
    
    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])
        
        # Pass SparseD_param for sparse/adaptive models
        if self.sparse_param is not None:
            # For loglikelihood tasks, we need to add 'now_step' which controls sparse warmup/scheduling.
            # Default keeps historical behavior (0 -> warmup -> standard attention).
            sparse_param_copy = self.sparse_param.copy()
            if 'now_step' not in sparse_param_copy:
                sparse_param_copy['now_step'] = int(self.likelihood_now_step) if self.likelihood_now_step is not None else 0
            if self.recompute_mask_each_call:
                sparse_param_copy['recompute_mask_each_call'] = True
            logits = self.model(batch, SparseD_param=sparse_param_copy).logits
        else:
            logits = self.model(batch).logits
        
        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]
    
    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        
        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)
            mask_indices = perturbed_seq == self.mask_id
            logits = self.get_logits(perturbed_seq, prompt_index)
            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())
        
        return - sum(loss_acc) / len(loss_acc)
    
    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False
        
        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix
        
        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)
            
            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        
        correct = target == seq[0, len(prefix):]
        return torch.all(correct)
    
    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]
        
        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]
        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]
        
        return context_enc, continuation_enc
    
    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }
        
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        
        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]
                ll = self.get_loglikelihood(prefix, target)
                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)
                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        
        torch.cuda.empty_cache()
        return out
    
    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
    
    def generate_until(self, requests):
        def _tokenize(e):
            return {
                "question": self.tokenizer(e["question"])["input_ids"],
                "question_text": e["question"],
                "until": e["until"],
            }
        
        ds = [{"question": req.args[0], "until": req.args[1]['until']} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        
        out = []
        for elem in tqdm(ds, desc="Generating..."):
            prompt = elem["question"].unsqueeze(0).to(self.device)
            stop_tokens = elem["until"]
            
            # Select appropriate generate function based on model type
            if self.model_type == 'standard':
                generated_answer = standard_generate(
                    self.model, prompt,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    temperature=0,
                    cfg_scale=self.cfg,
                    remasking=self.remasking,
                    mask_id=self.mask_id
                )
            elif self.model_type == 'sparse':
                generated_answer = sparse_generate(
                    self.model, prompt,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    temperature=0,
                    cfg_scale=self.cfg,
                    remasking=self.remasking,
                    mask_id=self.mask_id,
                    SparseD_param=self.sparse_param
                )
            elif self.model_type == 'adaptive':
                generated_answer = adaptive_generate(
                    self.model, prompt,
                    steps=self.steps,
                    gen_length=self.gen_length,
                    block_length=self.block_length,
                    temperature=0,
                    cfg_scale=self.cfg,
                    remasking=self.remasking,
                    mask_id=self.mask_id,
                    SparseD_param=self.sparse_param
                )
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")
            
            generated_answer = self.tokenizer.decode(generated_answer[0][prompt.shape[1]:], skip_special_tokens=False)
            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]
            
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            out.append(generated_answer)
            
            if self.accelerator:
                self.accelerator.wait_for_everyone()
        
        return out


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()

