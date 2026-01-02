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
        
        # NOTE:
        # Dream 的实现里，attention_mask 既可能是 "full"（表示不传 mask，等价于 attn_mask=None），
        # 也可能是 [B,1,N,N] 的 bool mask（见 generation_utils_dream.py）。
        # 这里不要把 "full" 变成 tensor，否则会把 SDPA 的 attn_mask 形状弄错，导致行为偏离真实生成。
        
        # Create shared position embeddings (matches DreamBaseModel.forward)
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        # Forward through layers
        for layer_idx, decoder_layer in enumerate(self.model.layers):
            if layer_idx == target_layer_idx:
                # 临时覆盖 self_attn.forward，注入指定的 head attention 输出；
                # 这样可以复用 DreamDecoderLayer.forward 的残差/MLP 等逻辑，尽量贴近真实 forward。
                original_attn_forward = decoder_layer.self_attn.forward

                def _attn_override(
                    hidden_states: torch.Tensor,
                    attention_mask: Optional[torch.Tensor] = None,
                    position_ids: Optional[torch.LongTensor] = None,
                    past_key_value=None,
                    output_attentions: bool = False,
                    use_cache: bool = False,
                    cache_position=None,
                    position_embeddings=None,
                ):
                    # head_att_values: (B, nh, T, hs) -> (B, T, hidden_size) -> o_proj -> (B, T, hidden_size)
                    attn_output = head_att_values.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
                    attn_output = decoder_layer.self_attn.o_proj(attn_output)
                    return attn_output, None, past_key_value

                decoder_layer.self_attn.forward = _attn_override  # type: ignore[method-assign]
                try:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )
                finally:
                    decoder_layer.self_attn.forward = original_attn_forward  # type: ignore[method-assign]
                hidden_states = layer_outputs[0]
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
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
        
        # Hook: capture SDPA output inside DreamSdpaAttention (pre-o_proj, shape: [B, nh, T, hs])
        captured_att = [None]
        original_sdpa = F.scaled_dot_product_attention

        def sdpa_with_capture(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
            out = original_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
            captured_att[0] = out.detach()
            return out

        # Create shared position embeddings (matches DreamBaseModel.forward)
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        # Forward through layers, patch SDPA only for the target layer
        for layer_idx, decoder_layer in enumerate(self.model.layers):
            if layer_idx == target_layer_idx:
                F.scaled_dot_product_attention = sdpa_with_capture
                try:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )
                    hidden_states = layer_outputs[0]
                finally:
                    F.scaled_dot_product_attention = original_sdpa
                break
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )
                hidden_states = layer_outputs[0]
        
        if was_training:
            self.model.train()
        
        if captured_att[0] is None:
            raise ValueError(f"Failed to capture attention at layer {target_layer_idx}")

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
        mask_token_id: Optional[int] = None,
        objective: str = "target_logprob",
        objective_margin_use_logprob: bool = False,
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

        def _objective_from_logits(logits: torch.Tensor) -> torch.Tensor:
            """
            Supported objectives:
            - "target_logprob" (default): sum log p(target)
            - "target_logit": sum logit(target)
            - "margin": sum (score(target) - max score(other)),
                        where score is logit by default, or logprob if objective_margin_use_logprob=True.
            """
            logits_at_changed = logits[:, changed_positions, :]  # (B, |pos|, V)
            target_tokens = actual_input_ids[:, changed_positions]  # (B, |pos|)

            if objective == "target_logit":
                return torch.gather(logits_at_changed, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1).sum()

            log_probs = F.log_softmax(logits_at_changed, dim=-1)
            if objective == "target_logprob":
                return torch.gather(log_probs, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1).sum()

            if objective == "margin":
                score = log_probs if objective_margin_use_logprob else logits_at_changed
                target_score = torch.gather(score, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)
                score_others = score.clone()
                score_others.scatter_(-1, target_tokens.unsqueeze(-1), float("-inf"))
                best_other = score_others.max(dim=-1).values
                return (target_score - best_other).sum()

            raise ValueError(f"Unknown objective={objective!r}. Use 'target_logprob', 'target_logit', or 'margin'.")
        
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
            
            # 标量目标（越大越好）
            output_scalar = _objective_from_logits(logits)
            
            # 计算梯度：∂output/∂att_interpolated
            grads = torch.autograd.grad(
                outputs=output_scalar,
                inputs=att_interpolated,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )[0]

            # 梯形法则权重
            weight = 0.5 if (step == 0 or step == self.n_steps) else 1.0
            accumulated_grads += grads * weight
        
        # Step 4: 归因 = (att_actual - att_baseline) ⊙ 平均梯度
        avg_grads = accumulated_grads / self.n_steps  # (B, nh, T, hs)
        
        # 逐元素乘法
        elementwise_attribution = (att_actual - att_baseline) * avg_grads  # (B, nh, T, hs)
        
        # 只在本 step 被修改的位置上聚合，避免 prompt/未修改 token 主导归因
        pos = torch.tensor(changed_positions, device=elementwise_attribution.device, dtype=torch.long)
        pos_attr = elementwise_attribution[:, :, pos, :]  # (B, nh, |pos|, hs)

        # 可选：只统计从 [MASK] -> token 的变化（如果提供 mask_token_id）
        if mask_token_id is not None:
            changed = (baseline_input_ids[:, pos] == mask_token_id).to(pos_attr.dtype)  # (B, |pos|)
            pos_attr = pos_attr * changed[:, None, :, None]

        attributions = pos_attr.sum(dim=[0, 2, 3])  # (nh,)
        
        return attributions
    
    def compute_full_generation_attribution(
        self,
        generation_history: List[torch.LongTensor],  # 每个 step 的序列状态
        mask_token_id: int,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        objective: str = "target_logprob",
        objective_margin_use_logprob: bool = False,
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
                    mask_token_id=mask_token_id,
                    objective=objective,
                    objective_margin_use_logprob=objective_margin_use_logprob,
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

