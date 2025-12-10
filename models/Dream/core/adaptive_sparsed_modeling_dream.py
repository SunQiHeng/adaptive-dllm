"""
Adaptive Sparsed Attention Modeling for Dream

This module implements adaptive sparse attention where different attention heads
have different sparsity levels based on their importance scores.

For GQA (Grouped Query Attention), different groups can have different sparsity levels.

Adapted from LLaDA implementation for Dream model architecture.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
import torch._dynamo

torch._dynamo.config.cache_size_limit = 10000
torch._dynamo.config.accumulated_cache_size_limit = 10000
flex_attn = torch.compile(flex_attention, dynamic=False)

from .sparsed_modeling_dream import (
    DreamAttention,
    DreamDecoderLayer,
    DreamBaseModel,
    DreamModel
)
from ..sparse.SparseD_utils_dream import create_block_mask_cached, customize_mask, create_attention_block_mask


class AdaptiveDreamAttention(DreamAttention):
    """
    Adaptive sparse attention for Dream.
    
    Extends SparsedDreamAttention to support per-head/per-group adaptive sparsity.
    Different attention heads/groups have different sparsity levels based on importance.
    """
    
    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        
        # Dream uses non-causal attention for diffusion generation
        self.is_causal = False
        
        # Adaptive sparsity attributes
        self.head_sparsity_levels = None  # Will be set by set_adaptive_sparsity
        self.head_importance_scores = None
        
        # Cache for sparse masks (unified for all heads)
        self.fine_mask = None
        self.block_mask = None
        self.last = None
    
    def set_adaptive_sparsity(
        self,
        sparsity_levels: torch.Tensor,
        importance_scores: Optional[torch.Tensor] = None
    ):
        """
        Set per-head/per-group adaptive sparsity levels.
        
        Args:
            sparsity_levels: Tensor of shape (n_heads,) or (n_kv_heads,) containing
                           keep ratios (fraction of tokens to KEEP, not drop)
            importance_scores: Optional importance scores for reference
        """
        n_kv_heads = self.config.num_key_value_heads
        n_heads = self.config.num_attention_heads
        
        if sparsity_levels.shape[0] == n_kv_heads:
            # Sparsity defined per KV head/group
            self.head_sparsity_levels = sparsity_levels
        elif sparsity_levels.shape[0] == n_heads:
            # Sparsity defined per query head, need to aggregate for KV heads
            heads_per_group = n_heads // n_kv_heads
            # Average sparsity across query heads in each group
            self.head_sparsity_levels = sparsity_levels.view(n_kv_heads, heads_per_group).mean(dim=1)
        else:
            raise ValueError(
                f"sparsity_levels shape {sparsity_levels.shape} doesn't match "
                f"n_heads={n_heads} or n_kv_heads={n_kv_heads}"
            )
        
        self.head_importance_scores = importance_scores
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        SparseD_param: Optional[Dict] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with adaptive sparse attention.
        """
        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings
        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = self._apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # Handle past key values for caching using Cache object
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
        # Apply adaptive sparse attention
        if SparseD_param is not None and self.head_sparsity_levels is not None:
            # For adaptive sparse attention, pass unrepeated k/v (n_kv_heads)
            attn_output = self._adaptive_sparse_attention(
                query_states, key_states, value_states, SparseD_param
            )
        else:
            # Repeat k/v heads if n_kv_heads < n_heads (GQA) for standard attention
            key_states = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states = self._repeat_kv(value_states, self.num_key_value_groups)
            
            # Fall back to standard attention
            # Dream uses non-causal attention for diffusion generation
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states, value_states,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                is_causal=False
            )
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        
        attn_output = self.o_proj(attn_output)
        
        return attn_output, None, past_key_value
    
    def _adaptive_sparse_attention(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        SparseD_param: Dict
    ) -> torch.Tensor:
        """
        Perform adaptive sparse attention with different sparsity per head/group.
        
        Args:
            query_states: Query tensor (B, n_heads, T, head_dim)
            key_states: Key tensor (B, n_kv_heads, T, head_dim) - unrepeated for GQA
            value_states: Value tensor (B, n_kv_heads, T, head_dim) - unrepeated for GQA
            SparseD_param: Sparsity parameters
        
        Returns:
            Attention output tensor (B, n_heads, T, head_dim)
        """
        now_step = SparseD_param['now_step']
        whole_steps = SparseD_param['whole_steps']
        new_generation = SparseD_param['new_generation']
        block_size = SparseD_param['block_size']
        
        B, n_heads, q_len, head_dim = query_states.shape
        _, n_kv_heads, kv_len, _ = key_states.shape
        
        # Initialize masks on first step
        if now_step == 0:
            self.fine_mask = None
            self.block_mask = None
            self.last = None
        
        # Get adaptive skip threshold
        skip_threshold = int(whole_steps * SparseD_param.get('skip', 0.2))
        end_time = skip_threshold + 1
        
        if now_step <= end_time:
            if now_step == end_time:
                # Build per-head adaptive masks (with unrepeated k/v)
                self._build_adaptive_masks(
                    query_states, key_states, value_states, 
                    block_size, new_generation, SparseD_param
                )
            
            # During warmup, repeat k/v for standard attention
            key_states_repeated = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states_repeated = self._repeat_kv(value_states, self.num_key_value_groups)
            
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states_repeated, value_states_repeated,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                is_causal=False
            )
        else:
            # Use adaptive sparse attention with unified block mask
            # Repeat k/v for flex_attn
            key_states_repeated = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states_repeated = self._repeat_kv(value_states, self.num_key_value_groups)
            
            attn_output = flex_attn(query_states, key_states_repeated, value_states_repeated, block_mask=self.block_mask)
        
        return attn_output
    
    def _build_adaptive_masks(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        block_size: int,
        new_generation: int,
        SparseD_param: Dict
    ):
        """
        Build adaptive sparse masks with different sparsity per head/group.
        
        Build masks for all heads at once but use different keep_ratio for each head.
        
        Args:
            query_states: (B, n_heads, q_len, head_dim)
            key_states: (B, n_kv_heads, kv_len, head_dim) - unrepeated for GQA
            value_states: (B, n_kv_heads, kv_len, head_dim) - unrepeated for GQA
        """
        B, n_heads, q_len, head_dim = query_states.shape
        _, n_kv_heads, kv_len, _ = key_states.shape
        
        # For GQA: map query heads to KV heads
        heads_per_group = n_heads // n_kv_heads
        
        # Create mask for ALL heads at once
        self.fine_mask = torch.zeros(
            (B, n_heads, (q_len+block_size-1)//block_size, (kv_len+block_size-1)//block_size),
            dtype=torch.bool, device=query_states.device
        )
        
        self.last = None
        
        # Build mask block by block for all heads
        for idx in range((q_len+block_size-1)//block_size):
            if q_len - idx*block_size <= new_generation or idx == (q_len+block_size-1)//block_size - 1:
                if self.last is None:
                    self.last = idx
            
            q_block = query_states[:, :, idx*block_size:(idx+1)*block_size, :]
            
            # Compute attention scores for all query heads with their corresponding KV heads
            # For GQA: each query head uses its corresponding KV head
            # q_block: (B, n_heads, block_len, head_dim)
            # key_states: (B, n_kv_heads, kv_len, head_dim)
            
            # Apply different keep_ratio for each head
            for head_idx in range(n_heads):
                # Get corresponding KV head
                kv_head_idx = head_idx // heads_per_group
                kv_head_idx = min(kv_head_idx, len(self.head_sparsity_levels) - 1)
                
                # Get base keep_ratio from adaptive config
                base_keep_ratio = self.head_sparsity_levels[kv_head_idx].item()
                
                # Apply select as multiplier: when select=1.0, use full adaptive ratio
                # when select<1.0, reduce sparsity proportionally across all heads
                select = SparseD_param.get('select', 1.0)
                keep_ratio = base_keep_ratio * select
                
                # Compute attention for this query head with its KV head
                q_head = q_block[:, head_idx:head_idx+1, :, :]  # (B, 1, block_len, head_dim)
                k_head = key_states[:, kv_head_idx:kv_head_idx+1, :, :]  # (B, 1, kv_len, head_dim)
                
                attn_weights = torch.matmul(q_head, k_head.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                
                # Create mask for this head with its specific keep_ratio
                head_mask = create_attention_block_mask(
                    attn_weights, block_size=block_size, keep_ratio=keep_ratio
                )
                self.fine_mask[:, head_idx:head_idx+1, idx:idx+1, :] = head_mask[:, :, :1, :]
        
        # Set last blocks to False
        if self.last is not None:
            self.fine_mask[:, :, :, self.last:] = False
        
        # Handle last blocks
        if self.last is not None:
            k_last = key_states[:, :, self.last*block_size:, :]
            for idx in range((q_len+block_size-1)//block_size):
                q_block = query_states[:, :, idx*block_size:(idx+1)*block_size, :]
                
                # Apply different keep_ratio for each head
                for head_idx in range(n_heads):
                    kv_head_idx = head_idx // heads_per_group
                    kv_head_idx = min(kv_head_idx, len(self.head_sparsity_levels) - 1)
                    
                    # Get base keep_ratio and apply select multiplier
                    base_keep_ratio = self.head_sparsity_levels[kv_head_idx].item()
                    select = SparseD_param.get('select', 1.0)
                    keep_ratio = base_keep_ratio * select
                    
                    # Compute attention for this query head with its KV head
                    q_head = q_block[:, head_idx:head_idx+1, :, :]
                    k_head_last = k_last[:, kv_head_idx:kv_head_idx+1, :, :]
                    
                    attn_weights = torch.matmul(q_head, k_head_last.transpose(2, 3)) / math.sqrt(head_dim)
                    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                    
                    head_mask = create_attention_block_mask(
                        attn_weights, block_size=block_size, keep_ratio=keep_ratio
                    )
                    self.fine_mask[:, head_idx:head_idx+1, idx:idx+1, self.last:] = torch.logical_or(
                        self.fine_mask[:, head_idx:head_idx+1, idx:idx+1, self.last:], 
                        head_mask[:, :, :1, :]
                    )
        
        # Convert to block mask for ALL heads at once
        new_mask = customize_mask(self.fine_mask, block_size=block_size)
        self.block_mask = create_block_mask_cached(
            new_mask, B, n_heads, q_len, kv_len, device=query_states.device, _compile=True
        )
    
    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        """Apply rotary position embedding."""
        # Simplified RoPE application
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed
    
    def _rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def _repeat_kv(self, hidden_states, n_rep):
        """Repeat key/value heads for GQA."""
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_key_value_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class AdaptiveDreamDecoderLayer(DreamDecoderLayer):
    """
    Adaptive sparse decoder layer for Dream.
    """
    
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        # Replace self_attn with adaptive version
        self.self_attn = AdaptiveDreamAttention(config, layer_idx)
    
    def set_adaptive_sparsity(self, sparsity_levels: torch.Tensor, importance_scores: Optional[torch.Tensor] = None):
        """Set adaptive sparsity for this layer's attention."""
        self.self_attn.set_adaptive_sparsity(sparsity_levels, importance_scores)


class AdaptiveDreamBaseModel(DreamBaseModel):
    """
    Adaptive sparse Dream base model (encoder layers without lm_head).
    """
    
    def __init__(self, config):
        super().__init__(config)
        # Replace layers with adaptive versions
        self.layers = nn.ModuleList(
            [AdaptiveDreamDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        
        # Adaptive config
        self.adaptive_config = None
        
        # Re-initialize to ensure adaptive layers are properly set up
        self.post_init()
    
    def set_adaptive_config(self, adaptive_config: Dict):
        """
        Set adaptive sparsity configuration for all layers.
        
        Args:
            adaptive_config: Dictionary containing:
                - importance_scores: Dict[int, torch.Tensor]
                - sparsity_levels: Dict[int, torch.Tensor]
                - metadata: Dict
        """
        self.adaptive_config = adaptive_config
        importance_scores = adaptive_config['importance_scores']
        sparsity_levels = adaptive_config['sparsity_levels']
        
        # Set sparsity for each layer
        for layer_idx, layer in enumerate(self.layers):
            if layer_idx in sparsity_levels:
                layer.set_adaptive_sparsity(
                    sparsity_levels[layer_idx],
                    importance_scores.get(layer_idx, None)
                )


class AdaptiveDreamModel(DreamModel):
    """
    Adaptive sparse Dream model for masked language modeling.
    Inherits from DreamModel which includes lm_head and diffusion_generate.
    """
    
    def __init__(self, config, adaptive_config: Optional[Dict] = None):
        # Initialize parent first (this creates self.model as DreamBaseModel)
        super().__init__(config)
        
        # Replace self.model with adaptive version
        self.model = AdaptiveDreamBaseModel(config)
        
        # Re-initialize weights for new layers
        self.post_init()
        
        # Set adaptive config if provided
        if adaptive_config is not None:
            self.model.set_adaptive_config(adaptive_config)
    
    def set_adaptive_config(self, adaptive_config: Dict):
        """
        Set adaptive sparsity configuration (proxy to self.model).
        
        Args:
            adaptive_config: Dictionary containing adaptive sparsity configuration
        """
        self.model.set_adaptive_config(adaptive_config)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, adaptive_config=None, **kwargs):
        """
        Load pretrained model with adaptive config.
        
        Args:
            pretrained_model_name_or_path: Model identifier
            adaptive_config: Dictionary containing adaptive sparsity configuration
            **kwargs: Additional arguments for model loading
        """
        # Remove adaptive_config from kwargs before passing to parent
        kwargs_copy = kwargs.copy()
        kwargs_copy.pop('adaptive_config', None)
        
        # Load standard DreamModel first
        from .sparsed_modeling_dream import DreamModel
        base_model = DreamModel.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs_copy)
        
        # Replace layers in base_model.model with adaptive layers IN-PLACE
        config = base_model.config
        new_layers = []
        
        for i, base_layer in enumerate(base_model.model.layers):
            # Create adaptive layer
            adaptive_layer = AdaptiveDreamDecoderLayer(config, i)
            
            # Copy weights from base layer to adaptive layer
            # The adaptive layer's self_attn is AdaptiveDreamAttention which inherits from DreamAttention
            # So we can safely load the state dict
            adaptive_layer.load_state_dict(base_layer.state_dict(), strict=False)
            
            # Move to the same device and dtype as the base layer
            device = base_layer.self_attn.q_proj.weight.device
            dtype = base_layer.self_attn.q_proj.weight.dtype
            adaptive_layer = adaptive_layer.to(device=device, dtype=dtype)
            
            new_layers.append(adaptive_layer)
        
        # Replace layers
        base_model.model.layers = nn.ModuleList(new_layers)
        
        # Change the class of base_model to AdaptiveDreamModel
        # This is a bit hacky but works for our purposes
        base_model.__class__ = cls
        base_model.model.__class__ = AdaptiveDreamBaseModel
        
        # Set adaptive config
        if adaptive_config is not None:
            base_model.model.set_adaptive_config(adaptive_config)
        
        return base_model


# Alias for compatibility
AdaptiveDreamForMaskedLM = AdaptiveDreamModel

