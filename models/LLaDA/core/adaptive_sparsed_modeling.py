"""
Adaptive Sparsed Attention Modeling for LLaDA

This module implements adaptive sparse attention where different attention heads/groups
have different sparsity levels based on their importance scores.

For GQA (Grouped Query Attention), different groups can have different sparsity levels.
"""

from __future__ import annotations

import math
import sys
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
import torch._dynamo

torch._dynamo.config.cache_size_limit = 10000
torch._dynamo.config.accumulated_cache_size_limit = 10000
flex_attn = torch.compile(flex_attention, dynamic=False)

from .sparsed_modeling import LLaDABlock, LLaDALlamaBlock, LLaDAModel, LLaDAModelLM
from ..sparse.sparsed_utils import create_block_mask_cached, customize_mask, create_attention_block_mask


class AdaptiveLLaDALlamaBlock(LLaDALlamaBlock):
    """
    Adaptive sparse attention block for LLaDA Llama architecture.
    
    Extends LLaDALlamaBlock to support per-head/per-group adaptive sparsity.
    Different attention heads/groups have different sparsity levels based on importance.
    """
    
    def __init__(self, layer_id: int, config, cache):
        super().__init__(layer_id, config, cache)
        
        # Adaptive sparsity attributes
        self.head_sparsity_levels = None  # Will be set by set_adaptive_sparsity
        # Optional per-query-head sparsity/weight vector (n_heads,). Used to avoid
        # averaging away head-level attribution signals under GQA.
        self.head_sparsity_levels_q = None
        self.head_importance_scores = None
        
        # Cache for sparse masks (unified for all heads, like standard sparse)
        self.fine_mask = None  # Unified mask for all heads
        self.block_mask = None  # Unified block mask for all heads
        self.last = None  # Last index
    
    def set_adaptive_sparsity(
        self,
        sparsity_levels: torch.Tensor,
        importance_scores: Optional[torch.Tensor] = None
    ):
        """
        Set per-head/per-group adaptive sparsity levels.
        
        Args:
            sparsity_levels: Tensor of shape (n_heads,) or (n_kv_heads,) containing
                           **relative weights** (mean≈1.0) OR absolute keep_ratios, depending on upstream config.
                           In our eval pipeline we typically use relative weights and convert to keep_ratio via:
                               keep_ratio = weight * select  (clamped to [min_keep_ratio, 1.0]).
                           IMPORTANT: When sparsity_levels are provided per *query head* (n_heads), we prefer
                           preserving per-Q-head resolution (even under GQA) to avoid averaging away head-level
                           attribution signals. If sparsity_levels are provided per KV head/group (n_kv_heads),
                           we use KV-level weights as before.
            importance_scores: Optional importance scores for reference
        """
        # For GQA, n_kv_heads < n_heads
        n_kv_heads = self.config.effective_n_kv_heads
        n_heads = self.config.n_heads
        
        if sparsity_levels.shape[0] == n_kv_heads:
            # Sparsity defined per KV head/group
            self.head_sparsity_levels = sparsity_levels
            self.head_sparsity_levels_q = None
        elif sparsity_levels.shape[0] == n_heads:
            # Sparsity/weights defined per query head.
            # Preserve per-Q-head resolution for adaptive masks.
            self.head_sparsity_levels_q = sparsity_levels
            # Also keep a KV-group summary for any legacy/debug usage.
            heads_per_group = n_heads // n_kv_heads
            self.head_sparsity_levels = sparsity_levels.view(n_kv_heads, heads_per_group).mean(dim=1)
        else:
            raise ValueError(
                f"sparsity_levels shape {sparsity_levels.shape} doesn't match "
                f"n_heads={n_heads} or n_kv_heads={n_kv_heads}"
            )
        
        self.head_importance_scores = importance_scores
        
        # Don't print here - will be noisy during model loading
        # The relative weights will be printed in a summary after model initialization
    
    def attention_adaptive(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        SparseD_param: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Adaptive sparse attention with per-head/per-group sparsity.
        
        Different heads/groups apply different sparsity levels based on their importance.
        """
        if SparseD_param is not None:
            now_step = SparseD_param['now_step']
            whole_steps = SparseD_param['whole_steps']
            new_generation = SparseD_param['new_generation']
            block_size = SparseD_param['block_size']
        
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype
        
        # Apply layer norm to keys and queries
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)
        
        # Reshape to separate heads
        # q: (B, T, n_heads, head_dim) -> (B, n_heads, T, head_dim)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # k, v: (B, T, n_kv_heads, head_dim) -> (B, n_kv_heads, T, head_dim)
        k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)
        
        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]
        
        if self.config.rope:
            q, k = self.rotary_emb(q, k)
        
        if attention_bias is not None:
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )
        
        # Apply adaptive sparse attention
        if SparseD_param is None or self.head_sparsity_levels is None:
            # Fall back to standard attention
            att = self._scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0 if not self.training else self.config.attention_dropout,
                is_causal=False,
            )
        else:
            # Adaptive sparse attention with per-head sparsity
            att = self._adaptive_sparse_attention(
                q, k, v, SparseD_param, block_size
            )
        
        # Re-assemble all head outputs
        att = att.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.attn_out(att), present
    
    def _adaptive_sparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        SparseD_param: Dict,
        block_size: int
    ) -> torch.Tensor:
        """
        Perform adaptive sparse attention with different sparsity per head/group.
        
        Args:
            q: Query tensor (B, n_heads, T, head_dim)
            k: Key tensor (B, n_kv_heads, T, head_dim)
            v: Value tensor (B, n_kv_heads, T, head_dim)
            SparseD_param: Sparsity parameters
            block_size: Block size for block-sparse attention
        
        Returns:
            Attention output tensor (B, n_heads, T, head_dim)
        """
        now_step = SparseD_param['now_step']
        whole_steps = SparseD_param['whole_steps']
        new_generation = SparseD_param['new_generation']
        
        B, n_heads, q_len, head_dim = q.shape
        _, n_kv_heads, kv_len, _ = k.shape

        # IMPORTANT:
        # For GQA, q has n_heads while k/v have n_kv_heads. flex_attention (and我们的 mask 构建逻辑)
        # 需要 head 维度对齐，否则会导致 head 对应关系错位甚至维度不广播。
        if n_heads != n_kv_heads:
            assert n_heads % n_kv_heads == 0
            repeat_factor = n_heads // n_kv_heads
            k_for_q = k.repeat_interleave(repeat_factor, dim=1, output_size=n_heads)
            v_for_q = v.repeat_interleave(repeat_factor, dim=1, output_size=n_heads)
        else:
            k_for_q = k
            v_for_q = v
        
        # Initialize masks on first step
        if now_step == 0:
            self.fine_mask = None
            self.block_mask = None
            self.last = None
        
        # Get adaptive skip threshold based on sparsity levels
        skip_threshold = int(whole_steps * SparseD_param.get('skip', 0.2))
        end_time = skip_threshold + 1
        
        if now_step <= end_time:
            if now_step == end_time:
                # Build per-head adaptive masks
                self._build_adaptive_masks(
                    q, k_for_q, v_for_q, block_size, new_generation, SparseD_param
                )
            
            # During warmup, use standard attention
            att = self._scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0.0 if not self.training else self.config.attention_dropout,
                is_causal=False,
            )
        else:
            # Use adaptive sparse attention with unified block mask
            att = flex_attn(q, k_for_q, v_for_q, block_mask=self.block_mask)
        
        return att
    
    def _build_adaptive_masks(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_size: int,
        new_generation: int,
        SparseD_param: Dict
    ):
        """
        Build adaptive sparse masks with different sparsity per head/group.
        
        Each head independently selects important blocks based on its own attention pattern.
        This preserves multi-head diversity (different heads can focus on different aspects).
        
        The select parameter controls the average keep_ratio across all heads, while maintaining
        the relative differences between heads based on their importance scores.
        
        head_sparsity_levels contains relative importance weights (mean=1.0) from
        allocate_adaptive_sparsity_from_importance. At inference, we multiply by select:
            keep_ratio = weight * select
        """
        B, n_heads, q_len, head_dim = q.shape
        _, _k_heads, kv_len, _ = k.shape
        # IMPORTANT:
        # - head_sparsity_levels is stored per KV head/group (n_kv_heads).
        # - head_sparsity_levels_q (if set) is stored per query head (n_heads) and takes precedence.
        # Even if k/v have been expanded to n_heads for computation, the mapping from q-head -> kv-head
        # must be based on the *original* effective_n_kv_heads from config, not k.shape[1].
        effective_n_kv_heads = int(self.config.effective_n_kv_heads)
        assert n_heads % effective_n_kv_heads == 0, "n_heads must be divisible by effective_n_kv_heads"
        heads_per_group = n_heads // effective_n_kv_heads
        
        # Get select parameter (controls average keep_ratio)
        select = SparseD_param.get('select', 1.0)
        if SparseD_param.get("debug_adaptive", False) and self.layer_id == 0:
            # Quick summary to verify adaptive config is actually being consumed.
            min_keep_ratio = float(SparseD_param.get("min_keep_ratio", 0.01))
            keep_by_kv = torch.clamp(self.head_sparsity_levels.to(torch.float32) * float(select), min=min_keep_ratio, max=1.0)
            print(
                f"[ADAPTIVE DEBUG] layer={self.layer_id} select={float(select):.4f} "
                f"keep_ratio(kv): mean={float(keep_by_kv.mean()):.4f} "
                f"std={float(keep_by_kv.std()):.4f} "
                f"min={float(keep_by_kv.min()):.4f} max={float(keep_by_kv.max()):.4f} "
                f"(n_kv={keep_by_kv.numel()}, n_q={n_heads})"
            )
        
        # head_sparsity_levels are relative weights (mean≈1.0)
        # Multiply by select to get actual keep_ratios
        # This ensures: mean(keep_ratio) ≈ select, while preserving relative importance
        
        # Create mask for ALL heads at once
        self.fine_mask = torch.zeros(
            (B, n_heads, (q_len+block_size-1)//block_size, (kv_len+block_size-1)//block_size),
            dtype=torch.bool, device=q.device
        )
        
        self.last = None
        
        # Build mask block by block for all heads
        for idx in range((q_len+block_size-1)//block_size):
            if q_len - idx*block_size <= new_generation or idx == (q_len+block_size-1)//block_size - 1:
                if self.last is None:
                    self.last = idx
            
            q_block = q[:, :, idx*block_size:(idx+1)*block_size, :]
            
            # Compute attention scores for all heads (k is expected to have n_heads here; see _adaptive_sparse_attention)
            attn_weights = torch.matmul(q_block, k.transpose(2, 3)) / math.sqrt(head_dim)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            
            # Apply different keep_ratio for each head
            for head_idx in range(n_heads):
                # Prefer per-query-head weights if available; otherwise fall back to KV-group weights.
                if getattr(self, "head_sparsity_levels_q", None) is not None:
                    relative_weight = self.head_sparsity_levels_q[head_idx].item()
                else:
                    # Get relative weight for this head's KV group (GQA: multiple q heads share one kv group)
                    kv_head_idx = head_idx // heads_per_group
                    # kv_head_idx should always be in-range now.
                    if kv_head_idx >= self.head_sparsity_levels.numel():
                        raise RuntimeError(
                            f"kv_head_idx out of range: kv_head_idx={kv_head_idx}, "
                            f"numel(head_sparsity_levels)={self.head_sparsity_levels.numel()}, "
                            f"n_heads={n_heads}, effective_n_kv_heads={effective_n_kv_heads}"
                        )
                    relative_weight = self.head_sparsity_levels[kv_head_idx].item()
                
                # Compute actual keep_ratio: weight * select
                # Important heads have weight>1.0, less important have weight<1.0
                # Average weight≈1.0, so average(keep_ratio)≈select
                keep_ratio = relative_weight * select
                # Clamp to a safe range to avoid empty masks
                min_keep_ratio = float(SparseD_param.get("min_keep_ratio", 0.01))
                keep_ratio = max(min(keep_ratio, 1.0), min_keep_ratio)
                
                # Create mask for this head with its specific keep_ratio
                head_attn = attn_weights[:, head_idx:head_idx+1, :, :]
                head_mask = create_attention_block_mask(
                    head_attn, block_size=block_size, keep_ratio=keep_ratio
                )
                self.fine_mask[:, head_idx:head_idx+1, idx:idx+1, :] = head_mask[:, :, :1, :]
        
        # Set last blocks to False
        self.fine_mask[:, :, :, self.last:] = False
        
        # Handle last blocks
        if self.last is not None:
            k_last = k[:, :, self.last*block_size:, :]
            for idx in range((q_len+block_size-1)//block_size):
                q_block = q[:, :, idx*block_size:(idx+1)*block_size, :]
                attn_weights = torch.matmul(q_block, k_last.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
                
                # Apply different keep_ratio for each head
                for head_idx in range(n_heads):
                    if getattr(self, "head_sparsity_levels_q", None) is not None:
                        relative_weight = self.head_sparsity_levels_q[head_idx].item()
                    else:
                        kv_head_idx = head_idx // heads_per_group
                        if kv_head_idx >= self.head_sparsity_levels.numel():
                            raise RuntimeError(
                                f"kv_head_idx out of range: kv_head_idx={kv_head_idx}, "
                                f"numel(head_sparsity_levels)={self.head_sparsity_levels.numel()}, "
                                f"n_heads={n_heads}, effective_n_kv_heads={effective_n_kv_heads}"
                            )
                        relative_weight = self.head_sparsity_levels[kv_head_idx].item()
                    
                    # Compute actual keep_ratio (same as above)
                    keep_ratio = relative_weight * select
                    min_keep_ratio = float(SparseD_param.get("min_keep_ratio", 0.01))
                    keep_ratio = max(min(keep_ratio, 1.0), min_keep_ratio)
                    
                    head_attn = attn_weights[:, head_idx:head_idx+1, :, :]
                    head_mask = create_attention_block_mask(
                        head_attn, block_size=block_size, keep_ratio=keep_ratio
                    )
                    self.fine_mask[:, head_idx:head_idx+1, idx:idx+1, self.last:] = torch.logical_or(
                        self.fine_mask[:, head_idx:head_idx+1, idx:idx+1, self.last:], 
                        head_mask[:, :, :1, :]
                    )
        
        # Convert to block mask for ALL heads at once
        new_mask = customize_mask(self.fine_mask, block_size=block_size)
        self.block_mask = create_block_mask_cached(
            new_mask, B, n_heads, q_len, kv_len, device=q.device, _compile=True
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        SparseD_param: Optional[Dict] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with adaptive sparse attention.
        """
        # Get query, key, value projections
        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)
        
        # Get attention scores with adaptive sparsity
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(
                self.attention_adaptive, q, k, v, attention_bias,
                layer_past=layer_past, use_cache=use_cache, SparseD_param=SparseD_param
            )
        else:
            att, cache = self.attention_adaptive(
                q, k, v, attention_bias,
                layer_past=layer_past, use_cache=use_cache, SparseD_param=SparseD_param
            )
        
        # Add attention scores
        x = x + self.dropout(att)
        
        # Add feed-forward projection
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)
        else:
            x = self.ff_norm(x)
        x, x_up = self.ff_proj(x), self.up_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)
        else:
            x = self.act(x)
        x = x * x_up
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x
        
        return x, cache


class AdaptiveLLaDAModel(LLaDAModel):
    """
    Adaptive sparse LLaDA model with per-head/per-group sparsity.
    """
    
    def __init__(self, config, init_params: bool = True, adaptive_config: Optional[Dict] = None):
        """
        Args:
            config: Model configuration
            init_params: Whether to initialize parameters
            adaptive_config: Adaptive sparsity configuration containing:
                - importance_scores: Per-layer head importance
                - sparsity_levels: Per-layer head sparsity levels
        """
        super().__init__(config, init_params)
        self.adaptive_config = adaptive_config
        
        # Replace blocks with adaptive blocks
        if adaptive_config is not None:
            self._convert_to_adaptive_blocks()
    
    def _convert_to_adaptive_blocks(self):
        """
        Convert standard blocks to adaptive blocks and set sparsity levels.
        """
        from .configuration import BlockType
        
        if self.config.block_type != BlockType.llama:
            print("Warning: Adaptive sparsity is currently only implemented for Llama blocks")
            return
        
        importance_scores = self.adaptive_config.get('importance_scores', {})
        sparsity_levels = self.adaptive_config.get('sparsity_levels', {})
        
        if self.config.block_group_size == 1:
            # Replace individual blocks
            new_blocks = []
            for block_idx, block in enumerate(self.transformer.blocks):
                # Create adaptive block
                adaptive_block = AdaptiveLLaDALlamaBlock(
                    block_idx, self.config, block._LLaDABlock__cache
                )
                # Copy weights from original block
                adaptive_block.load_state_dict(block.state_dict())
                
                # Set adaptive sparsity if available
                if block_idx in sparsity_levels:
                    adaptive_block.set_adaptive_sparsity(
                        sparsity_levels[block_idx],
                        importance_scores.get(block_idx)
                    )
                
                new_blocks.append(adaptive_block)
            
            self.transformer.blocks = nn.ModuleList(new_blocks)
            print(f"Converted {len(new_blocks)} blocks to adaptive blocks")
        else:
            print("Warning: Block groups not yet supported for adaptive sparsity")
    
    def set_adaptive_sparsity_config(self, adaptive_config: Dict):
        """
        Update adaptive sparsity configuration for all layers.
        """
        self.adaptive_config = adaptive_config
        
        importance_scores = adaptive_config.get('importance_scores', {})
        sparsity_levels = adaptive_config.get('sparsity_levels', {})
        
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if isinstance(block, AdaptiveLLaDALlamaBlock) and block_idx in sparsity_levels:
                    block.set_adaptive_sparsity(
                        sparsity_levels[block_idx],
                        importance_scores.get(block_idx)
                    )


class AdaptiveLLaDAModelLM(LLaDAModelLM):
    """
    Adaptive sparse LLaDA language model with per-head/per-group sparsity.
    """
    
    def __init__(self, config, model: Optional[AdaptiveLLaDAModel] = None, 
                 init_params: bool = False, adaptive_config: Optional[Dict] = None):
        # Don't call parent __init__ yet
        super(LLaDAModelLM, self).__init__(config)
        
        if not model:
            from .sparsed_modeling import create_model_config_from_pretrained_config
            model_config = create_model_config_from_pretrained_config(config)
            model_config.init_device = "cpu"
            self.model = AdaptiveLLaDAModel(model_config, init_params=init_params, 
                                           adaptive_config=adaptive_config)
        else:
            self.model = model
    
    def set_adaptive_sparsity_config(self, adaptive_config: Dict):
        """
        Update adaptive sparsity configuration.
        """
        if isinstance(self.model, AdaptiveLLaDAModel):
            self.model.set_adaptive_sparsity_config(adaptive_config)
        else:
            print("Warning: Model is not AdaptiveLLaDAModel, cannot set adaptive sparsity")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, adaptive_config=None, **kwargs):
        """
        Load pretrained model with adaptive config.
        
        Args:
            pretrained_model_name_or_path: Model identifier
            adaptive_config: Dictionary containing adaptive sparsity configuration
            **kwargs: Additional arguments for model loading
        
        Returns:
            AdaptiveLLaDAModelLM with adaptive sparsity applied
        """
        # Remove adaptive_config from kwargs before passing to parent
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('adaptive_config', None)
        
        # Load standard sparse LLaDAModelLM first
        from .sparsed_modeling import LLaDAModelLM as SparseLLaDAModelLM
        base_model = SparseLLaDAModelLM.from_pretrained(
            pretrained_model_name_or_path, *model_args, **kwargs_copy
        )
        
        # Get the internal LLaDAModel
        base_internal_model = base_model.model
        config = base_model.config
        
        # Store original device and dtype
        original_device = next(base_model.parameters()).device
        original_dtype = next(base_model.parameters()).dtype
        
        # Convert to AdaptiveLLaDAModel by changing class and converting blocks
        base_internal_model.__class__ = AdaptiveLLaDAModel
        base_internal_model.adaptive_config = adaptive_config
        
        # Convert blocks to adaptive blocks if adaptive_config is provided
        if adaptive_config is not None:
            base_internal_model._convert_to_adaptive_blocks()
            
            # Ensure all parameters have consistent dtype after conversion
            for param in base_model.parameters():
                if param.dtype != original_dtype:
                    param.data = param.data.to(original_dtype)
        
        # Change the wrapper class to AdaptiveLLaDAModelLM
        base_model.__class__ = cls
        
        return base_model

