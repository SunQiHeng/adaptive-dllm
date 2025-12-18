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


@register_model("dream_eval")
class DreamEvalHarness(LM):
    def __init__(
        self,
        model_path='/data/qh_models/Dream-v0-Instruct-7B',
        model_type='standard',  # 'standard', 'sparse', 'adaptive'
        mask_id=126336,
        max_length=4096,
        batch_size=32,
        mc_num=128,
        is_check_greedy=False,
        # Dream generation params (matching attribution settings)
        steps=32,
        max_new_tokens=256,
        temperature=0.8,
        top_p=0.9,
        top_k=None,
        eps=1e-3,
        alg='entropy',
        alg_temp=0.0,
        # Sparse params
        skip=0.2,
        select=0.3,
        block_size=128,
        # Adaptive params
        adaptive_config_path=None,
        importance_source='precomputed',  # 'precomputed', 'uniform', 'normal', or custom path
        base_sparsity=0.5,
        min_sparsity=0.1,
        max_sparsity=0.9,
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
            
            # Determine importance scores source
            from models.Dream.sparse.adaptive_utils_dream import create_adaptive_sparsity_config
            
            n_layers = self.model.config.num_hidden_layers
            n_heads = self.model.config.num_key_value_heads
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
                dream_importance_path = '/home/qiheng/Projects/adaptive-dllm/configs/head_importance_dream/head_importance.pt'
                if os.path.exists(dream_importance_path):
                    print(f"✓ Loading pre-computed importance scores from: {dream_importance_path}")
                    importance_data = torch.load(dream_importance_path, weights_only=False)
                    importance_scores = importance_data['importance_scores']
                else:
                    raise FileNotFoundError(f"Pre-computed importance file not found: {dream_importance_path}")
                
                adaptive_config = create_adaptive_sparsity_config(
                    n_layers=n_layers,
                    n_heads=n_heads,
                    importance_scores=importance_scores,  # Use precomputed
                    base_sparsity=base_sparsity,
                    min_sparsity=min_sparsity,
                    max_sparsity=max_sparsity,
                    seed=42
                )
                print(f"✓ Created adaptive config using PRE-COMPUTED importance scores")
            
            elif importance_source in ['uniform', 'normal', 'random']:
                # Generate random importance with specified distribution
                print(f"✓ Generating RANDOM importance scores with '{importance_source}' distribution")
                adaptive_config = create_adaptive_sparsity_config(
                    n_layers=n_layers,
                    n_heads=n_heads,
                    importance_scores=None,  # Generate random
                    strategy=importance_source,
                    base_sparsity=base_sparsity,
                    min_sparsity=min_sparsity,
                    max_sparsity=max_sparsity,
                    seed=42
                )
                print(f"✓ Created adaptive config using RANDOM '{importance_source}' importance")
            
            elif os.path.exists(importance_source):
                # Custom importance file path
                print(f"✓ Loading custom importance scores from: {importance_source}")
                importance_data = torch.load(importance_source, weights_only=False)
                importance_scores = importance_data['importance_scores']
                
                adaptive_config = create_adaptive_sparsity_config(
                    n_layers=n_layers,
                    n_heads=n_heads,
                    importance_scores=importance_scores,
                    base_sparsity=base_sparsity,
                    min_sparsity=min_sparsity,
                    max_sparsity=max_sparsity,
                    seed=42
                )
                print(f"✓ Created adaptive config using CUSTOM importance from: {importance_source}")
            
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
            
            self.sparse_param = {
                'skip': skip,
                'select': select,
                'block_size': block_size,
                'new_generation': max_new_tokens,
                'whole_steps': steps,
                'now_step': 0,  # Will be updated during generation
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
            mask_token_id=mask_id,
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
            print(f"Adaptive params: importance_source={importance_source}, base_sparsity={base_sparsity}, min={min_sparsity}, max={max_sparsity}")
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
    
    def apply_chat_template(self, chat_history, **kwargs):
        """Apply chat template to format conversations"""
        return self.tokenizer.apply_chat_template(
            chat_history, 
            tokenize=False, 
            **kwargs
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
    def get_logits(self, batch, prompt_index=None):
        """Get logits from model, with optional sparse parameters"""
        # For sparse/adaptive models, pass SparseD_param
        if self.sparse_param is not None:
            logits = self.model(batch, SparseD_param=self.sparse_param).logits
        else:
            logits = self.model(batch).logits
        
        return logits[:, :batch.shape[1]]
    
    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        """Compute log likelihood for Dream model"""
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
        """Check if greedy decoding matches target"""
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
        def _tokenize(e):
            # Tokenize the question directly (chat template should be applied by lm-eval if needed)
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
            
            # Use Dream's diffusion_generate method with generation config
            gen_config = self.generation_config
            
            # Generate using Dream's diffusion_generate method
            if self.sparse_param is not None:
                # For sparse/adaptive models, pass SparseD_param
                generated = self.model.diffusion_generate(
                    inputs=prompt,
                    generation_config=gen_config,
                    SparseD_param=self.sparse_param
                )
            else:
                # For standard model
                generated = self.model.diffusion_generate(
                    inputs=prompt,
                    generation_config=gen_config
                )
            
            # Decode the generated sequence
            if hasattr(generated, 'sequences'):
                generated_answer = generated.sequences[0]
            else:
                generated_answer = generated[0]
            
            # Remove prompt and decode
            generated_answer = self.tokenizer.decode(
                generated_answer[prompt.shape[1]:], 
                skip_special_tokens=False
            )
            
            # Apply stop tokens
            for stop_seq in stop_tokens:
                if stop_seq in generated_answer:
                    generated_answer = generated_answer.split(stop_seq)[0]
            
            # Re-tokenize and decode to clean up
            generated_answer_ids = self.tokenizer(generated_answer)["input_ids"]
            generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            out.append(generated_answer)
            
            if self.accelerator:
                self.accelerator.wait_for_everyone()
        
        return out


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()

