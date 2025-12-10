"""
Head Attribution using Integrated Gradients for Dream - Step-wise Version.

与LLaDA不同，Dream使用完全并行的refinement，没有block概念。
因此我们采用step-wise attribution：对每个diffusion step进行归因。

Step-wise attribution: 每个 diffusion step 完成后进行一次归因。
- Baseline: step i 开始时的状态
- Actual: step i 结束时的状态
- 归因目标: 该 step 中被修改的所有位置

Adapted for Dream's parallel refinement architecture.
"""
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Set


class StepwiseIntegratedGradientsAttributionDream:
    """
    Step-wise Integrated Gradients for Dream head attribution.
    
    核心思路：
    1. 每个 diffusion step 完成后进行一次归因
    2. Baseline: step 开始时的序列状态
    3. Actual: step 结束时的序列状态
    4. 归因目标：该 step 中实际被修改的位置（从[MASK]变为具体token）
    
    这适配Dream的并行refinement机制：
    - 每step同时处理多个位置
    - 不同step修改不同的位置集合
    - 基于confidence的调度
    """
    
    def __init__(
        self,
        model,
        n_steps: int = 10,
    ):
        """
        Args:
            model: DreamModel or DreamForMaskedLM instance
            n_steps: Integrated Gradients 的积分步数
        """
        # Handle both wrapper and direct model
        if hasattr(model, 'model'):
            self.model_wrapper = model
            self.model = model.model
        else:
            self.model_wrapper = None
            self.model = model
        
        self.n_steps = n_steps
        self.device = next(model.parameters()).device
        self.config = model.config
    
    def _forward_with_layer_head_cache(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        target_layer_idx: int,
        head_att_values: torch.Tensor,  # (B, nh, T, hs) - 该层各 head 的 att 值
    ) -> torch.Tensor:
        """
        Forward pass，使用给定的 head attention outputs 替换目标层的正常计算。
        
        Args:
            input_ids: Input token ids (B, L)
            attention_mask: Attention mask (B, L, L) or "full"
            position_ids: Position ids (B, L)
            target_layer_idx: 要替换 att 的层索引
            head_att_values: 要使用的 att 值 (B, nh, T, hs)
        
        Returns:
            logits: (B, L, vocab_size)
        """
        was_training = self.model.training
        self.model.eval()
        
        batch_size, seq_len = input_ids.shape
        hidden_size = self.config.hidden_size
        
        # Get embeddings
        hidden_states = self.model.embed_tokens(input_ids)
        
        # Prepare position_ids
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Prepare causal mask if attention_mask == "full"
        if isinstance(attention_mask, str) and attention_mask == "full":
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=hidden_states.device)
        
        # Forward through layers
        for layer_idx, decoder_layer in enumerate(self.model.layers):
            if layer_idx == target_layer_idx:
                # 使用给定的 att 值替换正常的 attention 计算
                residual = hidden_states
                hidden_states = decoder_layer.input_layernorm(hidden_states)
                
                # 直接使用 head_att_values 作为 attention 输出
                # head_att_values shape: (B, nh, T, hs)
                # 需要转换为 (B, T, hidden_size)
                attn_output = head_att_values.transpose(1, 2).contiguous()
                attn_output = attn_output.reshape(batch_size, seq_len, hidden_size)
                
                # 通过 o_proj
                attn_output = decoder_layer.self_attn.o_proj(attn_output)
                
                # Residual connection
                hidden_states = residual + attn_output
                
                # FFN
                residual = hidden_states
                hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
                hidden_states = decoder_layer.mlp(hidden_states)
                hidden_states = residual + hidden_states
            else:
                # 正常前向传播
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                hidden_states = layer_outputs[0]
        
        # Final layer norm
        hidden_states = self.model.norm(hidden_states)
        
        # Get logits
        if self.model_wrapper is not None and hasattr(self.model_wrapper, 'lm_head'):
            logits = self.model_wrapper.lm_head(hidden_states)
        else:
            logits = self.model.lm_head(hidden_states)
        
        if was_training:
            self.model.train()
        
        return logits
    
    def _compute_layer_head_att(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        target_layer_idx: int,
    ) -> torch.Tensor:
        """
        计算某一层的正常 attention outputs（在 attention 计算之后，o_proj 之前）。
        
        Returns:
            att: shape (B, nh, T, hs)
        """
        was_training = self.model.training
        self.model.eval()
        
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        hidden_states = self.model.embed_tokens(input_ids)
        
        # Prepare position_ids
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_states.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Prepare attention mask
        if isinstance(attention_mask, str) and attention_mask == "full":
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.bool, device=hidden_states.device)
        
        # Hook to capture attention output BEFORE reshape and o_proj
        # We need to capture at the point where shape is (B, nh, T, hs)
        captured_att = [None]
        
        def capture_attention_hook(module, input, output):
            # output[0] is the attention output AFTER o_proj: (B, T, hidden_size)
            # We need to recompute to get (B, nh, T, hs) before o_proj
            # So we can't use this hook - we'll use manual computation instead
            pass
        
        # Don't use hook - directly compute attention manually
        # Forward through layers up to (but not including) target layer
        target_layer = self.model.layers[target_layer_idx]
        
        for layer_idx, decoder_layer in enumerate(self.model.layers):
            if layer_idx < target_layer_idx:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                hidden_states = layer_outputs[0]
        
        # Always use manual computation to get (B, nh, T, hs)
        with torch.no_grad():
            residual = hidden_states
            hidden_states_norm = target_layer.input_layernorm(hidden_states)
            
            # Manually compute Q, K, V
            q = target_layer.self_attn.q_proj(hidden_states_norm)
            k = target_layer.self_attn.k_proj(hidden_states_norm)
            v = target_layer.self_attn.v_proj(hidden_states_norm)
            
            # Reshape to multi-head
            num_heads = self.config.num_attention_heads
            num_kv_heads = self.config.num_key_value_heads
            head_dim = self.config.hidden_size // num_heads
            
            q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
            k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            v = v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            
            # Apply RoPE
            cos, sin = target_layer.self_attn.rotary_emb(v, position_ids)
            q, k = self._apply_rotary_pos_emb(q, k, cos, sin)
            
            # Repeat k, v for GQA
            if num_kv_heads != num_heads:
                k = self._repeat_kv(k, num_heads // num_kv_heads)
                v = self._repeat_kv(v, num_heads // num_kv_heads)
            
            # Scaled dot-product attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=0.0,
                is_causal=True
            )
            
            captured_att[0] = attn_output  # (B, nh, T, hs)
        
        if was_training:
            self.model.train()
        
        return captured_att[0]
    
    def _apply_rotary_pos_emb(self, q, k, cos, sin):
        """Apply rotary position embedding."""
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
        batch, num_kv_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(
            batch, num_kv_heads, n_rep, slen, head_dim
        )
        return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)
    
    def compute_step_attribution_for_layer(
        self,
        baseline_input_ids: torch.LongTensor,
        actual_input_ids: torch.LongTensor,
        changed_positions: List[int],  # 该 step 中实际修改的位置
        target_layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        对某一层在某个 diffusion step 的归因。
        
        Args:
            baseline_input_ids: (B, L) - step 开始时的状态（修改前）
            actual_input_ids: (B, L) - step 结束时的状态（修改后）
            changed_positions: 该 step 中实际被修改的位置列表
            target_layer_idx: 要归因的层
            attention_mask: (B, L) or "full"
            position_ids: (B, L)
        
        Returns:
            attributions: shape (n_heads,) - 该层每个 head 在该 step 的重要性分数
        """
        if len(changed_positions) == 0:
            # 如果没有位置被修改，返回零归因
            num_heads = self.config.num_attention_heads
            return torch.zeros(num_heads, device=self.device)
        
        self.model.eval()
        num_heads = self.config.num_attention_heads
        
        # Step 1: 计算 baseline attention (step 开始时)
        with torch.no_grad():
            att_baseline = self._compute_layer_head_att(
                baseline_input_ids, attention_mask, position_ids, target_layer_idx
            )  # (B, nh, T, hs)
        
        # Step 2: 计算 actual attention (step 结束时)
        with torch.no_grad():
            att_actual = self._compute_layer_head_att(
                actual_input_ids, attention_mask, position_ids, target_layer_idx
            )  # (B, nh, T, hs)
        
        # Step 3: Integrated Gradients - 在 baseline 和 actual 之间插值
        accumulated_grads = torch.zeros_like(att_actual, device=self.device)  # (B, nh, T, hs)
        
        for step in range(self.n_steps + 1):
            alpha = step / self.n_steps
            
            # 插值：att_α = baseline + α * (actual - baseline)
            att_interpolated = att_baseline + alpha * (att_actual - att_baseline)
            att_interpolated = att_interpolated.detach().requires_grad_(True)
            
            # Forward pass 得到 logits（使用 step 结束时的 input_ids）
            logits = self._forward_with_layer_head_cache(
                input_ids=actual_input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                target_layer_idx=target_layer_idx,
                head_att_values=att_interpolated
            )
            
            # 提取被修改位置的 logits
            logits_at_changed = logits[:, changed_positions, :]  # (B, len(changed_positions), vocab_size)
            
            # 获取这些位置的真实 token
            target_tokens = actual_input_ids[:, changed_positions]  # (B, len(changed_positions))
            
            # 计算这些位置的 log probability 作为目标
            log_probs = F.log_softmax(logits_at_changed, dim=-1)
            target_log_probs = torch.gather(
                log_probs, 
                dim=-1, 
                index=target_tokens.unsqueeze(-1)
            ).squeeze(-1)  # (B, len(changed_positions))
            
            # 求和作为标量输出（越大越好）
            output_scalar = target_log_probs.sum()
            
            # 计算梯度：∂output/∂att_interpolated
            if att_interpolated.requires_grad:
                output_scalar.backward()
                
                grads = att_interpolated.grad
                
                # 梯形法则权重
                if step == 0 or step == self.n_steps:
                    weight = 0.5
                else:
                    weight = 1.0
                
                accumulated_grads += grads * weight
        
        # Step 4: 归因 = (att_actual - att_baseline) ⊙ 平均梯度
        avg_grads = accumulated_grads / self.n_steps  # (B, nh, T, hs)
        
        # 逐元素乘法
        elementwise_attribution = (att_actual - att_baseline) * avg_grads  # (B, nh, T, hs)
        
        # 对每个 head 在 batch, seq_len, head_dim 维度上求和
        attributions = elementwise_attribution.sum(dim=[0, 2, 3])  # (nh,)
        
        return attributions
    
    def compute_full_generation_attribution(
        self,
        generation_history: List[torch.LongTensor],  # 每个 step 的序列状态
        mask_token_id: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        对整个生成过程进行归因，累积所有 diffusion steps 的贡献。
        
        Args:
            generation_history: 列表，每个元素是一个 step 的序列状态 (B, L)
            mask_token_id: [MASK] token 的 ID
            attention_mask: (B, L) or "full"
            position_ids: (B, L)
        
        Returns:
            importance_scores: {layer_idx: attributions (n_heads,)} - 每层每个 head 的累积重要性
        """
        num_layers = len(self.model.layers)
        num_heads = self.config.num_attention_heads
        
        # 初始化累积归因
        importance_scores = {
            layer_idx: torch.zeros(num_heads, device=self.device)
            for layer_idx in range(num_layers)
        }
        
        # 遍历每个 diffusion step
        for step_idx in range(1, len(generation_history)):
            baseline_ids = generation_history[step_idx - 1]
            actual_ids = generation_history[step_idx]
            
            # 找出这一步中被修改的位置（从[MASK]变为具体token，或token被改变）
            changed_mask = (baseline_ids != actual_ids)
            
            # 只考虑从[MASK]变为非[MASK]的位置
            was_mask = (baseline_ids == mask_token_id)
            is_not_mask = (actual_ids != mask_token_id)
            valid_changes = changed_mask & was_mask & is_not_mask
            
            # 获取变化的位置索引
            changed_positions = torch.nonzero(valid_changes[0], as_tuple=True)[0].tolist()
            
            if len(changed_positions) == 0:
                continue
            
            print(f"Step {step_idx}: {len(changed_positions)} positions changed")
            
            # 对每一层计算归因
            for layer_idx in range(num_layers):
                layer_attribution = self.compute_step_attribution_for_layer(
                    baseline_input_ids=baseline_ids,
                    actual_input_ids=actual_ids,
                    changed_positions=changed_positions,
                    target_layer_idx=layer_idx,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                
                # 累积归因
                importance_scores[layer_idx] += layer_attribution
        
        return importance_scores
    
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

