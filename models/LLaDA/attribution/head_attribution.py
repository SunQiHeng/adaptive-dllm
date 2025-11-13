"""
Head Attribution using Integrated Gradients for Diffusion Language Models (LLaDA).

针对每一层的 attention heads 进行归因，baseline 是该层 att 完全 mask 掉。
"""
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np


def ensure_finite_(x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False):
    """
    Modify ``x`` in place to replace ``float("-inf")`` with the minimum value of the dtype when ``check_neg_inf``
    is ``True`` and to replace ``float("inf")`` with the maximum value of the dtype when ``check_pos_inf`` is ``True``.
    """
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)


class IntegratedGradientsHeadAttribution:
    """
    对 Diffusion LM 的每一层 attention heads 使用 Integrated Gradients 归因。
    
    核心思路：
    1. 逐层归因：对每一层单独计算 head importance
    2. Baseline: 该层所有 heads 的 att 输出都为 0
    3. Actual: 该层 heads 的正常 att 输出
    4. 在 baseline 和 actual 之间插值并积分
    """
    
    def __init__(
        self,
        model,
        n_steps: int = 20,
    ):
        """
        Args:
            model: LLaDAModel instance
            n_steps: Integrated Gradients 的积分步数
        """
        self.model = model
        self.n_steps = n_steps
        self.device = next(model.parameters()).device
    
    def _forward_with_layer_head_cache(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        target_layer_idx: int,
        head_att_values: torch.Tensor,  # (B, nh, T, hs) - 该层各 head 的 att 值
    ) -> torch.Tensor:
        """
        Forward pass，使用给定的 head attention outputs 替换目标层的正常计算。
        
        Args:
            input_ids: Input token ids (B, L)
            attention_mask: Attention mask (B, L)
            target_layer_idx: 要替换 att 的层索引
            head_att_values: 要使用的 att 值 (B, nh, T, hs)
        
        Returns:
            logits: (B, L, vocab_size)
        """
        # 确保模型处于 eval 模式（禁用 dropout）
        was_training = self.model.training
        self.model.eval()
        
        batch_size, seq_len = input_ids.shape
        d_model = self.model.config.d_model
        n_heads = self.model.config.n_heads
        head_dim = d_model // n_heads
        
        # Get embeddings
        x = self.model.transformer.wte(input_ids)
        
        # Input embedding processing
        if self.model.config.input_emb_norm:
            x = x * (d_model ** 0.5)
        
        if not (self.model.config.alibi or self.model.config.rope):
            pos = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            pos_emb = self.model.transformer.wpe(pos)
            x = pos_emb + x
        
        x = self.model.transformer.emb_drop(x)
        
        # Process attention mask and bias (following official implementation EXACTLY - line 1251-1292)
        # Transform the attention mask into what the blocks expect
        if attention_mask is not None and 0.0 in attention_mask:
            # shape: (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
        else:
            attention_mask = None
        
        # Merge attention mask with attention bias
        # 🔑 KEY: Only prepare attention_bias if needed (matching official logic in modeling_llada.py:1259-1273)
        attention_bias = None
        if (
            attention_mask is not None
            or self.model.config.alibi
            # NOTE: We don't have past_key_values in our use case
        ):
            if attention_bias is None and self.model.config.alibi:
                # Not implemented: would need get_causal_attention_bias + get_alibi_attention_bias
                raise NotImplementedError("ALiBi is not supported in this implementation")
            elif attention_bias is None:
                attention_bias = self.model.get_bidirectional_attention_bias(seq_len, x.device)
            
            # Transform to the right shape
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)
            
            # Add in the masking bias
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                # Handle -inf values
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)
        
        # Get blocks
        blocks = self.model.transformer.blocks if self.model.config.block_group_size == 1 else [
            block for group in self.model.transformer.block_groups for block in group
        ]
        
        # Forward through blocks
        for curr_layer_idx, block in enumerate(blocks):
            # Attention
            x_normed = block.attn_norm(x)
            
            if curr_layer_idx == target_layer_idx:
                # 使用给定的 att 值，不计算 attention
                # head_att_values 已经是 (B, nh, T, hs) 形状
                # 需要 merge heads: (B, nh, T, hs) -> (B, T, d_model)
                att = head_att_values.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
            else:
                # 正常计算 attention
                # Get Q, K, V
                if hasattr(block, 'att_proj'):
                    qkv = block.att_proj(x_normed)
                    head_dim_size = d_model // n_heads
                    fused_dims = (d_model, block.config.effective_n_kv_heads * head_dim_size, 
                                 block.config.effective_n_kv_heads * head_dim_size)
                    q, k, v = qkv.split(fused_dims, dim=-1)
                else:
                    q = block.q_proj(x_normed)
                    k = block.k_proj(x_normed)
                    v = block.v_proj(x_normed)
                
                # Reshape
                q = q.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
                k = k.view(batch_size, seq_len, block.config.effective_n_kv_heads, head_dim).transpose(1, 2)
                v = v.view(batch_size, seq_len, block.config.effective_n_kv_heads, head_dim).transpose(1, 2)
                
                # RoPE
                if block.config.rope:
                    q, k = block.rotary_emb(q, k)
                
                # GQA
                if k.size(1) != q.size(1):
                    k = k.repeat_interleave(n_heads // k.size(1), dim=1, output_size=n_heads)
                    v = v.repeat_interleave(n_heads // v.size(1), dim=1, output_size=n_heads)
                
                # Attention
                # Apply attention bias (slice for current sequence length)
                attn_bias_slice = attention_bias[:, :, :seq_len, :seq_len] if attention_bias is not None else None
                att = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_bias_slice,
                    dropout_p=0.0,
                    is_causal=False
                )
                
                # Merge heads: (B, nh, T, hs) -> (B, T, d_model)
                att = att.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
            
            # Apply output projection
            att = block.attn_out(att)
            x = x + block.dropout(att)
            
            # Feed-forward
            og_x = x
            x = block.ff_norm(x)
            
            if hasattr(block, 'att_proj'):
                x = block.ff_proj(x)
                x = block.act(x)
                x = block.ff_out(x)
            else:
                x_gate = block.ff_proj(x)
                x_up = block.up_proj(x)
                x = block.act(x_gate) * x_up
                x = block.ff_out(x)
            
            x = block.dropout(x)
            x = og_x + x
        
        # Final layer norm
        x = self.model.transformer.ln_f(x)
        
        # Get logits
        if self.model.config.weight_tying:
            logits = F.linear(x, self.model.transformer.wte.weight, None)
        else:
            logits = self.model.transformer.ff_out(x)
        
        # 恢复原始训练状态
        if was_training:
            self.model.train()
        
        return logits
    
    def _compute_layer_head_att(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        target_layer_idx: int,
    ) -> torch.Tensor:
        """
        计算某一层的正常 attention outputs（在 scaled_dot_product_attention 之后）。
        
        策略：使用 hook 捕获官方 block 的 attention 输出，确保与官方实现完全一致。
        
        Returns:
            att: shape (B, nh, T, hs)
        """
        was_training = self.model.training
        self.model.eval()
        
        batch_size, seq_len = input_ids.shape
        d_model = self.model.config.d_model
        
        # Get embeddings and process
        x = self.model.transformer.wte(input_ids)
        
        if self.model.config.input_emb_norm:
            x = x * (d_model ** 0.5)
        
        if not (self.model.config.alibi or self.model.config.rope):
            pos = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            pos_emb = self.model.transformer.wpe(pos)
            x = pos_emb + x
        
        x = self.model.transformer.emb_drop(x)
        
        # Process attention mask and bias (matching official logic)
        if attention_mask is not None and 0.0 in attention_mask:
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
        else:
            attention_mask = None
        
        attention_bias = None
        if (
            attention_mask is not None
            or self.model.config.alibi
        ):
            if attention_bias is None and self.model.config.alibi:
                raise NotImplementedError("ALiBi is not supported in this implementation")
            elif attention_bias is None:
                attention_bias = self.model.get_bidirectional_attention_bias(seq_len, x.device)
            
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)
            
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)
        
        blocks = self.model.transformer.blocks if self.model.config.block_group_size == 1 else [
            block for group in self.model.transformer.block_groups for block in group
        ]
        
        # 使用 hook 捕获目标层的 attention 输出
        captured_att = [None]
        
        # 替换 F.scaled_dot_product_attention 来捕获输出
        original_sdpa = F.scaled_dot_product_attention
        
        def sdpa_with_capture(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
            result = original_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
            captured_att[0] = result.clone()
            return result
        
        # Forward through blocks
        for curr_layer_idx, block in enumerate(blocks):
            if curr_layer_idx == target_layer_idx:
                # 在目标层，替换 SDPA 来捕获输出
                F.scaled_dot_product_attention = sdpa_with_capture
                try:
                    x, _ = block(x, attention_bias=attention_bias)
                finally:
                    F.scaled_dot_product_attention = original_sdpa
                break
            else:
                # 正常 forward
                x, _ = block(x, attention_bias=attention_bias)
        
        if was_training:
            self.model.train()
        
        if captured_att[0] is None:
            raise ValueError(f"Failed to capture attention at layer {target_layer_idx}")
        
        return captured_att[0]  # (B, nh, T, hs)
    
    def compute_head_attribution_for_layer(
        self,
        input_ids: torch.LongTensor,
        update_positions: List[int],
        target_layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        对某一层的所有 heads 使用 Integrated Gradients 归因。
        
        Args:
            input_ids: (1, L)
            update_positions: 被更新的位置列表，如 [10, 12, 13]
            target_layer_idx: 要归因的层
            attention_mask: (1, L)
        
        Returns:
            attributions: shape (n_heads,) - 该层每个 head 的重要性分数
        """
        self.model.eval()
        n_heads = self.model.config.n_heads
        batch_size = input_ids.shape[0]
        n_positions = len(update_positions)
        
        # Step 1: 计算该层正常的 att 输出 (actual)
        with torch.no_grad():
            att_actual = self._compute_layer_head_att(
                input_ids, attention_mask, target_layer_idx
            )  # (B, nh, T, hs)
        
        # Step 2: Baseline - 所有 heads 的 att 都为 0
        att_baseline = torch.zeros_like(att_actual)
        
        # Step 3: Integrated Gradients - 沿插值路径积分
        # 存储每个 head 在每个插值点的梯度（保持完整形状用于逐元素乘法）
        accumulated_grads = torch.zeros_like(att_actual, device=self.device)  # (B, nh, T, hs)
        
        for step in range(self.n_steps + 1):
            alpha = step / self.n_steps
            
            # 插值：所有 heads 一起从 baseline 到 actual
            # att_α = α * att_actual (因为 baseline = 0)
            att_interpolated = alpha * att_actual  # (B, nh, T, hs)
            att_interpolated = att_interpolated.detach().requires_grad_(True)
            
            # Forward pass 得到 logits
            logits = self._forward_with_layer_head_cache(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_layer_idx=target_layer_idx,
                head_att_values=att_interpolated
            )
            
            # 提取更新位置的 logits，并对目标位置求和作为标量输出
            logits_at_pos = logits[0, update_positions, :]  # (n_pos, vocab_size)
            
            # 对 logits 求和得到标量（用于计算梯度）
            # 这里我们对每个位置的 logits 的 L1 范数求和
            output_scalar = logits_at_pos.abs().sum()
            
            # 计算梯度：∂output/∂att_interpolated
            if att_interpolated.requires_grad:
                output_scalar.backward()
                
                # att_interpolated.grad shape: (B, nh, T, hs)
                grads = att_interpolated.grad
                
                # 梯形法则权重
                if step == 0 or step == self.n_steps:
                    weight = 0.5
                else:
                    weight = 1.0
                
                # 累积梯度（保持完整形状）
                accumulated_grads += grads * weight
        
        # Step 4: 归因 = (att_actual - att_baseline) ⊙ 平均梯度
        # 因为 baseline = 0，所以归因 = att_actual ⊙ 平均梯度
        # ⊙ 表示逐元素乘法（element-wise multiplication）
        
        # 计算平均梯度（保持完整形状）
        avg_grads = accumulated_grads / self.n_steps  # (B, nh, T, hs)
        
        # 逐元素乘法，然后对每个 head 求和
        # 这是标准的 Integrated Gradients 公式
        elementwise_attribution = att_actual * avg_grads  # (B, nh, T, hs)
        
        # 对每个 head 在 batch, seq_len, head_dim 维度上求和
        attributions = elementwise_attribution.sum(dim=[0, 2, 3])  # (nh,)
        
        return attributions
    
    def get_head_ranking(
        self,
        importance_scores: Dict[int, torch.Tensor]
    ) -> Dict[int, List[int]]:
        """
        获取每层 heads 的重要性排序。
        
        Returns:
            {layer_idx: [head_indices sorted by importance (descending)]}
        """
        rankings = {}
        for layer_idx, scores in importance_scores.items():
            sorted_indices = torch.argsort(scores, descending=True).cpu().tolist()
            rankings[layer_idx] = sorted_indices
        
        return rankings
