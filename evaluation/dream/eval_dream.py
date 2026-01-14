#!/usr/bin/env python3
"""
Dream Evaluation with lm-eval
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
import re
import json
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_CODE_FENCE_RE = re.compile(r"```(?:python)?\n([\s\S]*?)```", re.IGNORECASE)


def _maybe_extract_python_completion(context: str, completion: str) -> str:
    """
    Best-effort cleanup for code-completion style prompts (e.g., HumanEval).

    Why: Instruct models often emit explanations / markdown fences even when the task expects a raw
    function body continuation. For HumanEval (non-instruct), lm-eval's default post-processing does
    NOT strip those, causing near-zero pass@1.

    Heuristic (kept conservative):
    - Only triggers when the *context* looks like a Python function prompt (contains 'def ').
    - If a markdown code fence exists, extract the first fenced block.
    - Else if the completion contains a fence marker, truncate before it.
    """
    if "def " not in context:
        return completion

    m = _CODE_FENCE_RE.search(completion)
    if m:
        return m.group(1)

    if "```" in completion:
        return completion.split("```", 1)[0]

    return completion


def print_adaptive_config_stats(adaptive_config, select, n_layers, n_heads, model_name="Model"):
    """
    Print statistics about adaptive sparsity configuration.
    
    Args:
        adaptive_config: The adaptive configuration dict
        select: The select parameter (target average keep ratio)
        n_layers: Number of layers
        n_heads: Number of heads per layer in the adaptive config (can be Q-heads or KV-heads; inferred from data)
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
    print(f"Layers: {n_layers}, Heads per layer (config width): {n_heads}")
    print(f"Total heads: {n_layers * n_heads}")
    
    # Collect all weights/keep_ratios
    all_values = []
    layer_means = []
    
    print("\n" + "-" * 80)
    print("Per-Layer Statistics:")
    print("-" * 80)
    
    for layer_idx in range(n_layers):
        values = sparsity_levels[layer_idx]
        layer_mean = values.mean().item()
        layer_min = values.min().item()
        layer_max = values.max().item()
        
        all_values.append(values)
        layer_means.append(layer_mean)
        
        # Calculate actual keep_ratio if using relative weights
        if output_relative_weights:
            actual_mean = layer_mean * select
            actual_min = layer_min * select
            actual_max = layer_max * select
            # Clamp to [0, 1]
            actual_mean = min(actual_mean, 1.0)
            actual_max = min(actual_max, 1.0)
            
            print(f"Layer {layer_idx:2d}: weight_mean={layer_mean:.4f} "
                  f"→ keep_ratio={actual_mean:.4f} ({actual_mean*100:.1f}%), "
                  f"range=[{actual_min:.3f}, {actual_max:.3f}]")
        else:
            print(f"Layer {layer_idx:2d}: keep_ratio_mean={layer_mean:.4f} ({layer_mean*100:.1f}%), "
                  f"range=[{layer_min:.3f}, {layer_max:.3f}]")
    
    # Global statistics
    #
    # IMPORTANT (align with LLaDA):
    # - If output_relative_weights=True, values are *relative weights* (mean≈1.0), and actual keep_ratios are
    #   keep_ratio = clamp(weight * select, max=1.0).
    # - If output_relative_weights=False, values are already *absolute keep_ratios* in [0, 1].
    weights_tensor = torch.cat(all_values)
    if output_relative_weights:
        keep_tensor = torch.clamp(weights_tensor * float(select), max=1.0)
    else:
        keep_tensor = weights_tensor

    global_mean = keep_tensor.mean().item()
    global_min = keep_tensor.min().item()
    global_max = keep_tensor.max().item()
    global_std = keep_tensor.std().item()
    
    print("\n" + "-" * 80)
    print("Global Statistics:")
    print("-" * 80)
    
    if output_relative_weights:
        actual_global_mean = global_mean
        actual_global_min = global_min 
        actual_global_max = global_max
        # Clamp
        actual_global_mean = min(actual_global_mean, 1.0)
        actual_global_max = min(actual_global_max, 1.0)
        
        # Print weight stats separately (since global_* are keep_ratio stats here)
        w_mean = weights_tensor.mean().item()
        w_std = weights_tensor.std().item()
        w_min = weights_tensor.min().item()
        w_max = weights_tensor.max().item()
        print(f"Relative weights:")
        print(f"  Mean:   {w_mean:.4f} (should be ≈1.0)")
        print(f"  Std:    {w_std:.4f}")
        print(f"  Range:  [{w_min:.3f}, {w_max:.3f}]")
        print(f"\nActual keep_ratios (weights × select={select}):")
        print(f"  Mean:   {actual_global_mean:.4f} ({actual_global_mean*100:.1f}%)")
        print(f"  Target: {select:.4f} ({select*100:.1f}%)")
        print(f"  Deviation: {abs(actual_global_mean - select):.4f} ({abs(actual_global_mean - select)*100:.2f}%)")
        print(f"  Range:  [{actual_global_min:.3f}, {actual_global_max:.3f}]")
        
        # Count heads that will hit upper limit (keep_ratio > 1.0 after scaling)
        clamped_count = (weights_tensor * float(select) > 1.0).sum().item()
        total_heads = n_layers * n_heads
        print(f"  Heads hitting upper limit (>1.0): {clamped_count}/{total_heads} ({clamped_count/total_heads*100:.1f}%)")
        
        if abs(actual_global_mean - select) < 0.01:
            print(f"  ✅ Mean matches target (deviation < 1%)")
        elif abs(actual_global_mean - select) < 0.05:
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
        actual_layer_min = layer_mean_min * select
        actual_layer_max = layer_mean_max * select
        print(f"Layer means range: [{layer_mean_min:.4f}, {layer_mean_max:.4f}]")
        print(f"Layer means std:   {layer_mean_std:.4f}")
        print(f"Actual keep_ratio range across layers: [{actual_layer_min:.3f}, {actual_layer_max:.3f}]")
        print(f"Variation: {(actual_layer_max - actual_layer_min)*100:.1f}% spread")
    else:
        print(f"Layer means range: [{layer_mean_min:.4f}, {layer_mean_max:.4f}]")
        print(f"Layer means std:   {layer_mean_std:.4f}")
        print(f"Variation: {(layer_mean_max - layer_mean_min)*100:.1f}% spread")
    
    print("=" * 80 + "\n")


@register_model("dream_eval")
class DreamEvalHarness(LM):
    def __init__(
        self,
        model_path='/data/qh_models/Dream-v0-Instruct-7B',
        model_type='standard',  # 'standard', 'sparse', 'adaptive'
        mask_id=None,  # Will be set from tokenizer
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=False,
        # Dream generation params (FIXED to match official Dream eval)
        steps=32,
        max_new_tokens=256,
        temperature=0.1,  # Official default: 0.1
        top_p=0.9,
        top_k=None,
        eps=1e-3,
        alg='entropy',
        alg_temp=0.0,  # Official default: 0.0
        # Sparse params
        skip=0.2,
        select=0.3,
        block_size=128,
        # Adaptive params
        adaptive_config_path=None,
        importance_source='precomputed',  # 'precomputed', 'uniform', 'normal', or custom path
        precomputed_importance_path=None,
        gqa_weight_mode="kv",  # 'kv' (average within KV-group; GQA only, recommended default for Dream) | 'q' (per query head)
        relative_weight_scale=2.0 / 3.0,  # how far adaptive weights deviate from 1.0 (smaller => closer to uniform sparse)
        importance_increases_keep=True,   # True: higher importance -> keep more; False: invert (ablation)
        min_keep_ratio=0.01,              # safety clamp in adaptive mask building
        min_sparsity=0.1,
        max_sparsity=0.9,
        # Likelihood eval params (for multiple-choice tasks like MMLU)
        likelihood_now_step=None,         # Set > warmup to trigger sparse attention in likelihood scoring
        recompute_mask_each_call=False,  # Recompute masks for each forward (needed for dynamic sequences)
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
        # Note: Must use specific model classes (not AutoModel) to get generate() method
        print(f"\nLoading Dream model from {model_path}...")
        
        if model_type == 'standard':
            from models.Dream.core.modeling_dream import DreamModel
            self.model = DreamModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                **model_kwargs
            )
            self.sparse_param = None
            
        elif model_type == 'sparse':
            from models.Dream.core.sparsed_modeling_dream import DreamModel
            self.model = DreamModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                **model_kwargs
            )
            self.sparse_param = {
                'skip': skip,
                'select': select,
                'block_size': block_size,
                'new_generation': max_new_tokens,
                'whole_steps': steps,
                'now_step': 0,  # Will be updated during generation
            }
            
        elif model_type == 'adaptive':
            from models.Dream.core.adaptive_sparsed_modeling_dream import AdaptiveDreamModel
            import json
            
            self.model = AdaptiveDreamModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                **model_kwargs
            )
            
            # GQA sanity + default guidance:
            # Dream is typically a GQA model (num_key_value_heads < num_attention_heads). In that case,
            # applying weights per Q-head can be noisier than averaging within each KV group.
            try:
                n_q = int(getattr(self.model.config, "num_attention_heads", 0) or 0)
                n_kv = int(getattr(self.model.config, "num_key_value_heads", n_q) or n_q)
                if n_kv > 0 and n_q > 0 and (n_q % n_kv != 0):
                    raise ValueError(f"Invalid GQA config: num_attention_heads={n_q} not divisible by num_key_value_heads={n_kv}")
                if n_kv > 0 and n_kv < n_q and str(gqa_weight_mode) == "q":
                    print(
                        f"[warn] Dream config looks like GQA (num_attention_heads={n_q}, num_key_value_heads={n_kv}) "
                        f"but gqa_weight_mode='q'. If quality is poor, try gqa_weight_mode='kv' (group-average weights)."
                    )
            except Exception as e:
                print(f"[warn] Skipping GQA sanity check: {e}")
            
            # Determine importance scores source
            from models.Dream.sparse.adaptive_utils_dream import create_adaptive_sparsity_config
            
            n_layers = self.model.config.num_hidden_layers
            # Default to Q-heads for consistency with attribution outputs; may be overridden if we load a file.
            n_heads = self.model.config.num_attention_heads
            importance_scores = None
            
            # Option 1: Load from explicit config file path
            if adaptive_config_path and os.path.exists(adaptive_config_path):
                if adaptive_config_path.endswith('.json'):
                    with open(adaptive_config_path, 'r') as f:
                        adaptive_config = json.load(f)
                        if 'sparsity_levels' in adaptive_config:
                            sparsity_levels_dict = {}
                            for k, v in adaptive_config['sparsity_levels'].items():
                                sparsity_levels_dict[int(k)] = torch.tensor(v)
                            adaptive_config['sparsity_levels'] = sparsity_levels_dict
                else:
                    adaptive_config = torch.load(adaptive_config_path, weights_only=False)
                print(f"✓ Loaded adaptive config from: {adaptive_config_path}")
            
            # Option 2: Load from importance source specification
            elif importance_source == 'precomputed':
                # Use pre-computed importance scores from attribution
                dream_importance_path = precomputed_importance_path
                if os.path.exists(dream_importance_path):
                    print(f"✓ Loading pre-computed importance scores from: {dream_importance_path}")
                    importance_data = torch.load(dream_importance_path, weights_only=False)
                    importance_scores = importance_data['importance_scores']
                else:
                    raise FileNotFoundError(f"Pre-computed importance file not found: {dream_importance_path}")
                
                # Infer head dimension from file (can be Q-heads or KV-heads)
                try:
                    n_heads = int(importance_scores[0].numel())
                except Exception:
                    pass

                # Validate that head dimension matches model (Q-heads or KV-heads) to avoid silent misalignment.
                try:
                    n_q = int(getattr(self.model.config, "num_attention_heads", 0) or 0)
                    n_kv = int(getattr(self.model.config, "num_key_value_heads", n_q) or n_q)
                    if n_heads not in (n_q, n_kv):
                        raise ValueError(
                            f"Importance head dimension mismatch: file_heads={n_heads}, model_q_heads={n_q}, model_kv_heads={n_kv}. "
                            f"Expected file_heads to match Q-heads or KV-heads."
                        )
                except Exception as e:
                    raise RuntimeError(f"Invalid importance file for this model: {e}") from e
                
                adaptive_config = create_adaptive_sparsity_config(
                    n_layers=n_layers,
                    n_heads=n_heads,
                    importance_scores=importance_scores,  # Use precomputed
                    min_sparsity=min_sparsity,
                    max_sparsity=max_sparsity,
                    normalize_strategy='global_percentile',
                    output_relative_weights=True,
                    importance_increases_keep=bool(importance_increases_keep),
                    relative_weight_scale=float(relative_weight_scale),
                    seed=42
                )
                print(f"✓ Created adaptive config using PRE-COMPUTED importance scores")
                
                # Print detailed statistics
                print_adaptive_config_stats(adaptive_config, select, n_layers, n_heads, "Dream")
            
            elif importance_source in ['uniform', 'normal', 'random']:
                # Generate random importance with specified distribution
                print(f"✓ Generating RANDOM importance scores with '{importance_source}' distribution")
                # For Dream (GQA), generate per Q-head weights by default; attention module will aggregate to KV heads.
                n_heads = self.model.config.num_attention_heads
                adaptive_config = create_adaptive_sparsity_config(
                    n_layers=n_layers,
                    n_heads=n_heads,
                    importance_scores=None,  # Generate random
                    strategy=importance_source,
                    min_sparsity=min_sparsity,
                    max_sparsity=max_sparsity,
                    normalize_strategy='global_percentile',
                    output_relative_weights=True,
                    importance_increases_keep=bool(importance_increases_keep),
                    relative_weight_scale=float(relative_weight_scale),
                    seed=42
                )
                print(f"✓ Created adaptive config using RANDOM '{importance_source}' importance")
                
                # Print detailed statistics
                print_adaptive_config_stats(adaptive_config, select, n_layers, n_heads, "Dream")
            
            elif os.path.exists(importance_source):
                # Custom importance file path
                print(f"✓ Loading custom importance scores from: {importance_source}")
                importance_data = torch.load(importance_source, weights_only=False)
                importance_scores = importance_data['importance_scores']
                try:
                    n_heads = int(importance_scores[0].numel())
                except Exception:
                    pass
                
                adaptive_config = create_adaptive_sparsity_config(
                    n_layers=n_layers,
                    n_heads=n_heads,
                    importance_scores=importance_scores,
                    min_sparsity=min_sparsity,
                    max_sparsity=max_sparsity,
                    normalize_strategy='global_percentile',
                    output_relative_weights=True,
                    importance_increases_keep=bool(importance_increases_keep),
                    relative_weight_scale=float(relative_weight_scale),
                    seed=42
                )
                print(f"✓ Created adaptive config using CUSTOM importance from: {importance_source}")
                
                # Print detailed statistics
                print_adaptive_config_stats(adaptive_config, select, n_layers, n_heads, "Dream")
            
            else:
                raise ValueError(
                    f"Invalid importance_source: {importance_source}. "
                    f"Must be 'precomputed', 'uniform', 'normal', 'random', or a valid file path."
                )
            
            # Apply adaptive sparsity config to each layer
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
                sparsity_levels = adaptive_config.get('sparsity_levels', {})
                for layer_idx, layer in enumerate(self.model.model.layers):
                    if layer_idx in sparsity_levels and hasattr(layer, 'self_attn'):
                        if hasattr(layer.self_attn, 'set_adaptive_sparsity'):
                            keep_ratios = sparsity_levels[layer_idx]
                            importance = adaptive_config.get('importance_scores', {}).get(layer_idx, None)
                            layer.self_attn.set_adaptive_sparsity(keep_ratios, importance)
            
            print("\n" + "!" * 80)
            print("📌 重要说明：Adaptive Sparsity 工作原理")
            print("!" * 80)
            print("adaptive_config 中存储的是相对权重 (relative weights)，全局平均=1.0")
            print("在推理时，实际的 keep_ratio 计算方式为：")
            print(f"  keep_ratio = relative_weight × select = relative_weight × {select}")
            print("\n例如：")
            print(f"  如果某个head的 relative_weight = 1.2")
            print(f"  那么实际 keep_ratio = 1.2 × {select} = {1.2 * select:.3f} ({1.2 * select * 100:.1f}%)")
            print(f"  如果某个head的 relative_weight = 0.8")
            print(f"  那么实际 keep_ratio = 0.8 × {select} = {0.8 * select:.3f} ({0.8 * select * 100:.1f}%)")
            print("\n所有heads的实际keep_ratio的平均值 = {:.3f} ({}%)".format(select, select * 100))
            print("!" * 80 + "\n")
            
            self.sparse_param = {
                'skip': skip,
                'select': select,
                'block_size': block_size,
                'new_generation': max_new_tokens,
                'whole_steps': steps,
                'now_step': 0,  # Will be updated during generation
                'adaptive': True,
                # For GQA models only: controls whether to apply head weights per query head ('q')
                # or averaged per KV-group ('kv'). Default 'kv' is usually more stable for Dream.
                'gqa_weight_mode': str(gqa_weight_mode),
                # Safety clamp to avoid empty masks when a head gets very low weight.
                'min_keep_ratio': float(min_keep_ratio),
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

        # Print concrete CUDA device info for reproducibility/debugging.
        # Important: CUDA_VISIBLE_DEVICES can remap physical GPU ids.
        try:
            cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")
            if self.device.type == "cuda" and torch.cuda.is_available():
                cur = torch.cuda.current_device()
                name = torch.cuda.get_device_name(cur)
                print(f"[device] CUDA_VISIBLE_DEVICES={cuda_visible} | torch.cuda.current_device()={cur} | name={name}")
            else:
                print(f"[device] device={self.device} | CUDA_VISIBLE_DEVICES={cuda_visible}")
        except Exception as e:
            print(f"[device] (warn) failed to print CUDA device info: {e}")

        # Print key library versions for reproducibility.
        try:
            import platform
            import transformers
            try:
                import lm_eval  # type: ignore
                lm_eval_ver = getattr(lm_eval, "__version__", None)
            except Exception:
                lm_eval_ver = None
            try:
                import numpy as _np
                np_ver = getattr(_np, "__version__", None)
            except Exception:
                np_ver = None
            print(
                f"[env] python={platform.python_version()} | torch={torch.__version__} | "
                f"transformers={getattr(transformers, '__version__', None)} | lm_eval={lm_eval_ver} | numpy={np_ver}"
            )
        except Exception as e:
            print(f"[env] (warn) failed to print version info: {e}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # Get correct mask_id from tokenizer
        if mask_id is None:
            self.mask_id = self.tokenizer.mask_token_id if hasattr(self.tokenizer, 'mask_token_id') else 151666
        else:
            self.mask_id = mask_id
        
        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy
        self.likelihood_now_step = likelihood_now_step
        self.recompute_mask_each_call = recompute_mask_each_call
        
        # Dream generation config
        from models.Dream.generation_utils.generation_utils_dream import DreamGenerationConfig
        self.generation_config = DreamGenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            eps=eps,
            steps=steps,
            alg=alg,
            alg_temp=alg_temp,
            mask_token_id=self.mask_id,  # Use self.mask_id which was set from tokenizer
            pad_token_id=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else None,
            bos_token_id=self.tokenizer.bos_token_id if hasattr(self.tokenizer, 'bos_token_id') else None,
            eos_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None,
        )
        
        print(f"\n{'='*70}")
        print(f"Dream Evaluation Setup")
        print(f"{'='*70}")
        print(f"Model type: {model_type}")
        print(f"Model path: {model_path}")
        print(f"Steps: {steps}, Max new tokens: {max_new_tokens}")
        print(f"Temperature: {temperature}, Top-p: {top_p}, Top-k: {top_k}")
        if model_type in ['sparse', 'adaptive']:
            print(f"Sparse params: skip={skip}, select={select}, block_size={block_size}")
        if model_type == 'adaptive':
            print(f"Adaptive params: importance_source={importance_source}, min={min_sparsity}, max={max_sparsity}")
        print(f"{'='*70}\n")
    
    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size
    
    @property
    def tokenizer_name(self):
        """Return tokenizer name for chat template support"""
        return self.tokenizer.name_or_path if hasattr(self.tokenizer, 'name_or_path') else "dream_tokenizer"
    
    def apply_chat_template(self, chat_history, add_generation_prompt: bool = True, **kwargs):
        """
        Apply chat template to format conversations.

        Important: For *generation* we must end the prompt with an "open" assistant turn
        (i.e., include the assistant prefix but do NOT close it with the end token),
        otherwise many chat tokenizers will cause the model to immediately emit EOS.

        This mirrors Dream official eval implementation (`dream_official/.../diffllm.py`).
        """
        return self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            # If we are adding a generation prompt, we should NOT continue the final message.
            continue_final_message=not add_generation_prompt,
            **kwargs,
        )
    
    def _forward_process(self, batch, prompt_index):
        """Dream-specific forward process for perturbing sequences"""
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
    def get_logits(self, batch, prompt_index=None, attention_mask=None, position_ids=None):
        """Get logits from model, with optional sparse parameters"""
        # Construct attention_mask if not provided
        if attention_mask is None:
            # Use bool mask to avoid SDPA dtype issues (align with Dream attribution fixes).
            attention_mask = torch.ones_like(batch, dtype=torch.bool)
        
        # Construct position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(batch.shape[1], dtype=torch.long, device=batch.device).unsqueeze(0).expand(batch.shape[0], -1)
        
        # For sparse/adaptive models, pass SparseD_param
        if self.sparse_param is not None:
            # For loglikelihood tasks (MMLU), need to set now_step to trigger sparse attention
            sparse_param_copy = self.sparse_param.copy()
            # NOTE:
            # Dream's `self.sparse_param` is initialized with now_step=0, so checking
            # `'now_step' not in sparse_param_copy` will *never* override it.
            # For likelihood scoring we want to force now_step > warmup when requested.
            if self.likelihood_now_step is not None:
                sparse_param_copy['now_step'] = int(self.likelihood_now_step)
            if self.recompute_mask_each_call:
                sparse_param_copy['recompute_mask_each_call'] = True
            
            logits = self.model(
                input_ids=batch,
                attention_mask=attention_mask,
                position_ids=position_ids,
                SparseD_param=sparse_param_copy
            ).logits
        else:
            logits = self.model(
                input_ids=batch,
                attention_mask=attention_mask,
                position_ids=position_ids
            ).logits
        
        return logits[:, :batch.shape[1]]
    
    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        """Compute log likelihood for Dream model"""
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        
        # Construct attention_mask and position_ids for the sequence
        attention_mask = torch.ones_like(seq, dtype=torch.bool)
        position_ids = torch.arange(seq.shape[1], dtype=torch.long, device=self.device).unsqueeze(0).expand(seq.shape[0], -1)
        
        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)
            mask_indices = perturbed_seq == self.mask_id
            # Pass attention_mask and position_ids to get_logits
            logits = self.get_logits(perturbed_seq, prompt_index, attention_mask, position_ids)
            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())
        
        return - sum(loss_acc) / len(loss_acc)
    
    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        """Check if greedy decoding matches target"""
        if not self.is_check_greedy:
            return False
        
        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix
        
        # Construct attention_mask and position_ids for the sequence
        attention_mask = torch.ones_like(seq, dtype=torch.bool)
        position_ids = torch.arange(seq.shape[1], dtype=torch.long, device=self.device).unsqueeze(0)
        
        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            # Pass attention_mask and position_ids to get_logits
            logits = self.get_logits(seq, prompt_index, attention_mask, position_ids)[mask_index]
            x0 = torch.argmax(logits, dim=-1)
            
            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        
        correct = target == seq[0, len(prefix):]
        return torch.all(correct)
    
    def _encode_pair(self, context, continuation):
        """Encode context and continuation separately"""
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
        """Compute log likelihood for multiple requests"""
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
        """Generate text until stop tokens"""
        out = []
        
        for req in tqdm(requests, desc="Generating..."):
            # Get prompt and generation kwargs
            context = req.args[0]
            gen_kwargs = req.args[1]
            
            # DEBUG: Print first request context
            if len(out) == 0:
                print(f"\n[DEBUG] First request context:")
                print(f"  Context length: {len(context)} chars")
                print(f"  Context (first 300 chars): {repr(context[:300])}")
                print(f"  Context (last 200 chars): {repr(context[-200:])}")
                print(f"  gen_kwargs: {gen_kwargs}")
            
            # Tokenize prompt
            prompt_ids = self.tokenizer(context, return_tensors="pt").input_ids
            prompt_ids = prompt_ids.to(self.device)
            
            # Create attention mask
            attention_mask = torch.ones_like(prompt_ids, dtype=torch.bool)
            
            # Generate using Dream's diffusion_generate method
            # NOTE: Following official demo - pass first arg as positional, rest as kwargs
            if self.sparse_param is not None:
                # For sparse/adaptive models, pass SparseD_param
                generated = self.model.diffusion_generate(
                    prompt_ids,  # First arg positional (not inputs=)
                    attention_mask=attention_mask,
                    max_new_tokens=self.generation_config.max_new_tokens,
                    output_history=False,
                    return_dict_in_generate=True,
                    steps=self.generation_config.steps,
                    temperature=self.generation_config.temperature,
                    top_p=self.generation_config.top_p,
                    top_k=self.generation_config.top_k,
                    alg=self.generation_config.alg,
                    alg_temp=self.generation_config.alg_temp,
                    SparseD_param=self.sparse_param
                )
            else:
                # For standard model  
                generated = self.model.diffusion_generate(
                    prompt_ids,  # First arg positional (not inputs=)
                    attention_mask=attention_mask,
                    max_new_tokens=self.generation_config.max_new_tokens,
                    output_history=False,
                    return_dict_in_generate=True,
                    steps=self.generation_config.steps,
                    temperature=self.generation_config.temperature,
                    top_p=self.generation_config.top_p,
                    top_k=self.generation_config.top_k,
                    alg=self.generation_config.alg,
                    alg_temp=self.generation_config.alg_temp,
                )
            
            # Decode the generated sequence
            # NOTE: Following official - decode from prompt length onwards
            generated_tokens = generated.sequences[0]
            prompt_len = len(prompt_ids[0])
            
            # DEBUG: Print generation info for first request
            if len(out) == 0:
                print(f"\n[DEBUG] First generation:")
                print(f"  Prompt length: {prompt_len}")
                print(f"  Generated sequence length: {len(generated_tokens)}")
                print(f"  Generated tokens shape: {generated_tokens.shape}")
                print(f"  First 10 tokens: {generated_tokens[:10].tolist()}")
                print(f"  Last 10 tokens: {generated_tokens[-10:].tolist()}")
                print(f"  Mask token ID: {self.tokenizer.mask_token_id}")
            
            # Decode generated part (skip prompt)
            generated_ids = generated_tokens[prompt_len:].tolist()
            generated_answer = self.tokenizer.decode(generated_ids)
            
            # Split by EOS if present
            if self.tokenizer.eos_token in generated_answer:
                generated_answer = generated_answer.split(self.tokenizer.eos_token)[0]
            
            # Apply stop tokens from gen_kwargs
            stop_tokens = gen_kwargs.get('until', [])
            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]

            # Extra cleanup for code-completion tasks (notably HumanEval with instruct models)
            generated_answer = _maybe_extract_python_completion(context, generated_answer)
            
            # DEBUG: Print decoded answer for first request
            if len(out) == 0:
                print(f"  Generated answer length: {len(generated_answer)}")
                print(f"  Generated answer (first 200 chars): {repr(generated_answer[:200])}\n")
            
            out.append(generated_answer)
            
            if self.accelerator:
                self.accelerator.wait_for_everyone()
        
        return out


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()

