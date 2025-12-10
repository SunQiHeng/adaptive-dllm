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
        base_sparsity=0.5,
        min_sparsity=0.1,
        max_sparsity=0.9,
        importance_strategy='uniform',
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
            
            # Load adaptive config
            if adaptive_config_path and os.path.exists(adaptive_config_path):
                # Check file extension to determine load method
                if adaptive_config_path.endswith('.json'):
                    with open(adaptive_config_path, 'r') as f:
                        adaptive_config = json.load(f)
                    # Convert keep_mask to tensor
                    if 'keep_mask' in adaptive_config:
                        adaptive_config['keep_mask'] = torch.tensor(adaptive_config['keep_mask'], dtype=torch.bool)
                else:
                    adaptive_config = torch.load(adaptive_config_path, weights_only=False)
                print(f"Loaded adaptive config from {adaptive_config_path}")
            else:
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                n_layers = config.n_layers
                n_heads = config.n_kv_heads
                
                adaptive_config = create_adaptive_sparsity_config(
                    n_layers=n_layers,
                    n_heads=n_heads,
                    strategy=importance_strategy,
                    base_sparsity=base_sparsity,
                    min_sparsity=min_sparsity,
                    max_sparsity=max_sparsity,
                    seed=42
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
            # For loglikelihood tasks, we need to add 'now_step' which is used in generation
            # Set it to 0 since we're not doing incremental generation here
            sparse_param_copy = self.sparse_param.copy()
            if 'now_step' not in sparse_param_copy:
                sparse_param_copy['now_step'] = 0
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

