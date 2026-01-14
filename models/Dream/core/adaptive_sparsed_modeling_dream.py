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
                           **relative weights** (mean≈1.0) OR absolute keep_ratios, depending on upstream config.
                           In our eval pipeline we use relative weights (output_relative_weights=True) and convert
                           to actual keep_ratio via: keep_ratio = weight * select (clamped to [0, 1]).
                           IMPORTANT: For Dream (GQA), we still build sparse masks **per query head**, so if
                           sparsity_levels are provided per query head (n_heads), we should preserve that
                           resolution instead of averaging into KV groups.
            importance_scores: Optional importance scores for reference
        """
        n_kv_heads = self.config.num_key_value_heads
        n_heads = self.config.num_attention_heads
        
        if sparsity_levels.shape[0] == n_kv_heads:
            # Sparsity defined per KV head/group
            self.head_sparsity_levels = sparsity_levels
            self.head_sparsity_levels_q = None
        elif sparsity_levels.shape[0] == n_heads:
            # Sparsity/weights defined per *query head*.
            # Keep per-Q-head resolution for adaptive masks.
            self.head_sparsity_levels_q = sparsity_levels
            # Also keep a KV-group summary for any legacy paths (not used for mask building).
            if n_kv_heads <= 0 or n_heads % n_kv_heads != 0:
                raise ValueError(f"Invalid GQA config: n_heads={n_heads} not divisible by n_kv_heads={n_kv_heads}")
            heads_per_group = n_heads // n_kv_heads
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
        
        # NOTE:
        # - In likelihood scoring we may set `now_step` directly > warmup. In that case, we won't pass through
        #   the `now_step == 0` initialization below, so we must ensure these attributes exist.
        if not hasattr(self, "fine_mask"):
            self.fine_mask = None
        if not hasattr(self, "last"):
            self.last = None
        if not hasattr(self, "block_mask"):
            self.block_mask = None
        if not hasattr(self, "prev_seq_len"):
            self.prev_seq_len = None
        
        # Initialize masks on first step of each generation
        # CRITICAL: Must reset for each new sample as sequence length varies!
        if now_step == 0:
            self.fine_mask = None
            self.block_mask = None
            self.last = None
            self.prev_seq_len = None
            if self.layer_idx == 0:
                print(f"[Layer 0] Step 0 RESET: q_len={q_len}")
        
        # For likelihood evaluation we may want to recompute masks for every forward.
        # (Masks depend on the actual q/k content; caching across different examples is incorrect.)
        if SparseD_param.get("recompute_mask_each_call", False):
            self.fine_mask = None
            self.block_mask = None
            self.last = None
            self.prev_seq_len = None
        
        # Also reset if sequence length changed (new sample with different prompt length)
        if self.fine_mask is not None and hasattr(self, 'prev_seq_len') and self.prev_seq_len != q_len:
            if self.layer_idx == 0:
                print(f"[Layer 0] SEQ LENGTH CHANGED! prev={self.prev_seq_len}, new={q_len}, now_step={now_step}")
            self.fine_mask = None
            self.block_mask = None
            self.last = None
            self.prev_seq_len = None
        
        # Get adaptive skip threshold
        skip_threshold = int(whole_steps * SparseD_param.get('skip', 0.2))
        end_time = skip_threshold + 1
        
        # Adaptive sparse attention logic
        if now_step <= end_time:
            # Warmup phase: use standard attention
            if now_step == end_time:
                # Build per-head adaptive masks at the end of warmup
                self._build_adaptive_masks(
                    query_states, key_states, value_states, 
                    block_size, new_generation, SparseD_param
                )
                # Record sequence length for this generation
                self.prev_seq_len = q_len
                if self.layer_idx == 0:
                    mask_shape = self.fine_mask.shape if self.fine_mask is not None else None
                    print(f"[Layer 0] Step {now_step} BUILT MASK: q_len={q_len}, mask_shape={mask_shape}")
            
            # Use standard attention during warmup
            key_states_repeated = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states_repeated = self._repeat_kv(value_states, self.num_key_value_groups)
            
            attn_output = F.scaled_dot_product_attention(
                query_states, key_states_repeated, value_states_repeated,
                dropout_p=0.0 if not self.training else self.attention_dropout,
                is_causal=False
            )
        else:
            # Use adaptive sparse attention with unified block mask
            # Build masks on-demand if we didn't pass through the warmup "build" step.
            if self.block_mask is None:
                self._build_adaptive_masks(
                    query_states, key_states, value_states,
                    block_size, new_generation, SparseD_param
                )
            
            # Repeat k/v for flex_attn
            key_states_repeated = self._repeat_kv(key_states, self.num_key_value_groups)
            value_states_repeated = self._repeat_kv(value_states, self.num_key_value_groups)
            
            # `flex_attn` is torch.compile'd and may be unstable under highly dynamic shapes
            # (e.g., likelihood evaluation with recompute_mask_each_call). Fall back to eager
            # `flex_attention` in that case to avoid inductor/triton autotune crashes.
            use_compiled_flex = not SparseD_param.get("recompute_mask_each_call", False)
            if use_compiled_flex:
                attn_output = flex_attn(query_states, key_states_repeated, value_states_repeated, block_mask=self.block_mask)
            else:
                attn_output = flex_attention(query_states, key_states_repeated, value_states_repeated, block_mask=self.block_mask)
        
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
        The select parameter controls the average keep_ratio across all heads, while maintaining
        the relative differences between heads based on their importance scores.
        
        head_sparsity_levels contains relative importance weights (mean=1.0) from
        allocate_adaptive_sparsity_from_importance. At inference, we multiply by select:
            keep_ratio = weight * select
        
        Args:
            query_states: (B, n_heads, q_len, head_dim)
            key_states: (B, n_kv_heads, kv_len, head_dim) - unrepeated for GQA
            value_states: (B, n_kv_heads, kv_len, head_dim) - unrepeated for GQA
        """
        B, n_heads, q_len, head_dim = query_states.shape
        _, n_kv_heads, kv_len, _ = key_states.shape
        
        # For GQA: map query heads to KV heads (only needed if we only have KV-level weights)
        heads_per_group = n_heads // n_kv_heads
        
        # Get select parameter (controls average keep_ratio)
        select = float(SparseD_param.get('select', 1.0))
        min_keep_ratio = float(SparseD_param.get("min_keep_ratio", 0.01))
        gqa_weight_mode = str(SparseD_param.get("gqa_weight_mode", "kv"))
        if gqa_weight_mode not in ("q", "kv"):
            raise ValueError(f"Invalid gqa_weight_mode={gqa_weight_mode!r}. Expected 'q' or 'kv'.")
        
        # head_sparsity_levels are relative weights (mean≈1.0)
        # Multiply by select to get actual keep_ratios

        # Precompute per-(query-)head keep_ratio on-device.
        # NOTE: Avoid `.item()` in per-head loops: it triggers GPU↔CPU sync and is extremely slow.
        if getattr(self, "head_sparsity_levels_q", None) is not None and gqa_weight_mode == "q":
            relative_weights_q = self.head_sparsity_levels_q.to(device=query_states.device, dtype=torch.float32)
        else:
            relative_weights_kv = self.head_sparsity_levels.to(device=query_states.device, dtype=torch.float32)
            relative_weights_q = relative_weights_kv.repeat_interleave(heads_per_group, dim=0)[:n_heads]

        keep_ratio_per_head = torch.clamp(relative_weights_q * float(select), min=min_keep_ratio, max=1.0)  # (n_heads,)
        
        # Repeat KV heads to match Q heads once (GQA) - avoid doing this in every loop iteration
        k_for_q = key_states.repeat_interleave(heads_per_group, dim=1)[:, :n_heads, :, :]  # (B, n_heads, kv_len, head_dim)
        
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

            # Vectorize attention score computation across all heads.
            attn_weights = torch.matmul(q_block, k_for_q.transpose(2, 3)) / math.sqrt(head_dim)  # (B, n_heads, block_len, kv_len)
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # Vectorized per-head keep_ratio mask selection (no Python head loop).
            head_mask = create_attention_block_mask(attn_weights, block_size=block_size, keep_ratio=keep_ratio_per_head)
            self.fine_mask[:, :, idx:idx+1, :] = head_mask[:, :, :1, :]
        
        # Set last blocks to False
        if self.last is not None:
            self.fine_mask[:, :, :, self.last:] = False
        
        # Handle last blocks
        if self.last is not None:
            k_last = key_states[:, :, self.last*block_size:, :]
            k_last_for_q = k_last.repeat_interleave(heads_per_group, dim=1)[:, :n_heads, :, :]  # (B, n_heads, kv_len_last, head_dim)
            
            for idx in range((q_len+block_size-1)//block_size):
                q_block = query_states[:, :, idx*block_size:(idx+1)*block_size, :]
                
                # Vectorized attention with KV tail blocks.
                attn_weights = torch.matmul(q_block, k_last_for_q.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

                head_mask = create_attention_block_mask(attn_weights, block_size=block_size, keep_ratio=keep_ratio_per_head)
                self.fine_mask[:, :, idx:idx+1, self.last:] = torch.logical_or(
                    self.fine_mask[:, :, idx:idx+1, self.last:],
                    head_mask[:, :, :1, :],
                )
        
        # Convert to block mask for ALL heads at once
        new_mask = customize_mask(self.fine_mask, block_size=block_size)
        compile_masks = not SparseD_param.get("recompute_mask_each_call", False)
        self.block_mask = create_block_mask_cached(
            new_mask, B, n_heads, q_len, kv_len, device=query_states.device, _compile=compile_masks
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

