"""
Head Attribution using Integrated Gradients for Diffusion Language Models (LLaDA) - Version 2.

Block-wise attribution: 每个 block 完成后进行一次归因。
- Baseline: block 开始时的 attention 矩阵
- Actual: block 结束时的 attention 矩阵
"""
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional


def ensure_finite_(x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False):
    """
    Modify ``x`` in place to replace ``float("-inf")`` with the minimum value of the dtype when ``check_neg_inf``
    is ``True`` and to replace ``float("inf")`` with the maximum value of the dtype when ``check_pos_inf`` is ``True``.
    """
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)


class BlockwiseIntegratedGradientsAttribution:
    """
    Block-wise Integrated Gradients for LLaDA head attribution.
    
    核心思路：
    1. 每个 block 完成生成后进行一次归因
    2. Baseline: block 开始时的 attention 输出（x_start）
    3. Actual: block 结束时的 attention 输出（x_end）
    4. 归因目标：该 block 对应的生成位置的 logits
    """
    
    def __init__(
        self,
        model,
        n_steps: int = 10,
    ):
        """
        Args:
            model: LLaDAModel or LLaDAModelLM instance
            n_steps: Integrated Gradients 的积分步数
        """
        # Handle both LLaDAModelLM (has .model attr) and LLaDAModel (direct)
        if hasattr(model, 'model'):
            # LLaDAModelLM wrapper
            self.model_wrapper = model
            self.model = model.model
        else:
            # Direct LLaDAModel
            self.model_wrapper = None
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
        
        # NOTE:
        # LLaDA(MDM) 的 attention 实现（见 core/modeling.py）最终会在 block 内部以 attn_mask=None 调用
        # F.scaled_dot_product_attention（即忽略外部 attention_bias / attn_mask）。
        # 这里如果我们额外引入 bidirectional bias，会让归因 forward 与真实生成 forward 不一致。
        attention_bias = None
        
        # Get blocks
        blocks = self.model.transformer.blocks if self.model.config.block_group_size == 1 else [
            block for group in self.model.transformer.block_groups for block in group
        ]
        
        # Forward through blocks
        for curr_layer_idx, block in enumerate(blocks):
            if curr_layer_idx == target_layer_idx:
                # 临时覆盖 block.attention，把该层 attention 输出替换为给定 head_att_values。
                # 尽量复用原 block 的 LN/FFN/残差/dropout 逻辑，减少与真实 forward 的偏差。
                original_attention = block.attention

                def _attention_override(q, k, v, attention_bias=None, layer_past=None, use_cache=False):
                    att = head_att_values.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
                    return block.attn_out(att), None

                block.attention = _attention_override  # type: ignore[method-assign]
                try:
                    x, _ = block(x, attention_bias=attention_bias)
                finally:
                    block.attention = original_attention  # type: ignore[method-assign]
            else:
                x, _ = block(x, attention_bias=attention_bias)
        
        # Final layer norm
        x = self.model.transformer.ln_f(x)
        
        # Get logits
        if self.model.config.weight_tying:
            logits = F.linear(x, self.model.transformer.wte.weight, None)
        else:
            logits = self.model.transformer.ff_out(x)
        
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
        
        # Process attention mask and bias
        if attention_mask is not None and 0.0 in attention_mask:
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
        else:
            attention_mask = None
        
        attention_bias = None
        if attention_mask is not None or self.model.config.alibi:
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
        original_sdpa = F.scaled_dot_product_attention
        
        def sdpa_with_capture(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
            result = original_sdpa(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal)
            captured_att[0] = result.clone()
            return result
        
        # Forward through blocks
        for curr_layer_idx, block in enumerate(blocks):
            if curr_layer_idx == target_layer_idx:
                F.scaled_dot_product_attention = sdpa_with_capture
                try:
                    x, _ = block(x, attention_bias=attention_bias)
                finally:
                    F.scaled_dot_product_attention = original_sdpa
                break
            else:
                x, _ = block(x, attention_bias=attention_bias)
        
        if was_training:
            self.model.train()
        
        if captured_att[0] is None:
            raise ValueError(f"Failed to capture attention at layer {target_layer_idx}")
        
        return captured_att[0]  # (B, nh, T, hs)
    
    def compute_block_attribution_for_layer(
        self,
        baseline_input_ids: torch.LongTensor,
        actual_input_ids: torch.LongTensor,
        block_positions: List[int],
        target_layer_idx: int,
        attention_mask: Optional[torch.Tensor] = None,
        mask_id: int = 126336,
        objective: str = "target_logprob",
        objective_margin_use_logprob: bool = False,
        debug: bool = False,
        debug_topk: int = 5,
    ) -> torch.Tensor:
        """
        对某一层在某个 block 的归因。
        
        Args:
            baseline_input_ids: (B, L) - block 开始时的状态（如 x_0 或 x_s）
            actual_input_ids: (B, L) - block 结束时的状态（如 x_{s-1} 或 x_{2s-1}）
            block_positions: 该 block 对应的生成位置列表
            target_layer_idx: 要归因的层
            attention_mask: (B, L)
        
        Returns:
            attributions: shape (n_heads,) - 该层每个 head 在该 block 的重要性分数
        """
        self.model.eval()
        n_heads = self.model.config.n_heads

        def _objective_from_logits(logits: torch.Tensor) -> torch.Tensor:
            """
            Return scalar objective over this block positions, using actual_input_ids as targets.

            Supported objectives:
            - "target_logprob" (default): sum log p(target)
            - "target_logit": sum logit(target)
            - "margin": sum (score(target) - max score(other)), where score is logit by default,
                        or logprob if objective_margin_use_logprob=True.
            """
            logits_at_block = logits[:, block_positions, :]  # (B, |pos|, V)
            target_tokens = actual_input_ids[:, block_positions]  # (B, |pos|)

            if objective == "target_logit":
                return torch.gather(logits_at_block, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1).sum()

            log_probs = F.log_softmax(logits_at_block, dim=-1)
            if objective == "target_logprob":
                return torch.gather(log_probs, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1).sum()

            if objective == "margin":
                score = log_probs if objective_margin_use_logprob else logits_at_block
                target_score = torch.gather(score, dim=-1, index=target_tokens.unsqueeze(-1)).squeeze(-1)  # (B, |pos|)
                # Mask out target token to get best alternative
                score_others = score.clone()
                score_others.scatter_(-1, target_tokens.unsqueeze(-1), float("-inf"))
                ensure_finite_(score_others, check_neg_inf=True)
                best_other = score_others.max(dim=-1).values  # (B, |pos|)
                return (target_score - best_other).sum()

            raise ValueError(f"Unknown objective={objective!r}. Use 'target_logprob', 'target_logit', or 'margin'.")

        def _standard_forward_logits(input_ids: torch.LongTensor) -> torch.Tensor:
            """Run model forward without any attention override. Best-effort to extract logits."""
            with torch.no_grad():
                if self.model_wrapper is not None:
                    out = self.model_wrapper(input_ids, attention_mask=attention_mask)
                else:
                    out = self.model(input_ids, attention_mask=attention_mask)
            if hasattr(out, "logits"):
                return out.logits
            if isinstance(out, (tuple, list)) and len(out) > 0:
                return out[0]
            if torch.is_tensor(out):
                return out
            raise TypeError(f"Unexpected model output type for logits: {type(out)}")
        
        # Step 1: 计算 baseline attention (block 开始时)
        with torch.no_grad():
            att_baseline = self._compute_layer_head_att(
                baseline_input_ids, attention_mask, target_layer_idx
            )  # (B, nh, T, hs)
        
        # Step 2: 计算 actual attention (block 结束时)
        with torch.no_grad():
            att_actual = self._compute_layer_head_att(
                actual_input_ids, attention_mask, target_layer_idx
            )  # (B, nh, T, hs)

        if debug:
            # 1) IG-path baseline/actual (input_ids 固定为 actual_input_ids，只改 head_att_values)
            with torch.no_grad():
                logits_a0 = self._forward_with_layer_head_cache(
                    input_ids=actual_input_ids,
                    attention_mask=attention_mask,
                    target_layer_idx=target_layer_idx,
                    head_att_values=att_baseline,
                )
                logits_a1 = self._forward_with_layer_head_cache(
                    input_ids=actual_input_ids,
                    attention_mask=attention_mask,
                    target_layer_idx=target_layer_idx,
                    head_att_values=att_actual,
                )
                f_a0 = _objective_from_logits(logits_a0)
                f_a1 = _objective_from_logits(logits_a1)
                print(
                    f"[ATTR-DEBUG][layer={target_layer_idx}] IG-path f(a0)={float(f_a0):.6f} "
                    f"f(a1)={float(f_a1):.6f} Δ={float(f_a1 - f_a0):+.6f}"
                )

            # 2) Standard baseline_ids vs actual_ids (完全不 override attention)
            logits_std_base = _standard_forward_logits(baseline_input_ids)
            logits_std_act = _standard_forward_logits(actual_input_ids)
            f_std_base = _objective_from_logits(logits_std_base)
            f_std_act = _objective_from_logits(logits_std_act)
            print(
                f"[ATTR-DEBUG][layer={target_layer_idx}] STD f(base_ids)={float(f_std_base):.6f} "
                f"f(actual_ids)={float(f_std_act):.6f} Δ={float(f_std_act - f_std_base):+.6f}"
            )
        
        # Step 3: Integrated Gradients - 在 baseline 和 actual 之间插值
        accumulated_grads = torch.zeros_like(att_actual, device=self.device)  # (B, nh, T, hs)
        
        for step in range(self.n_steps + 1):
            alpha = step / self.n_steps
            
            # 插值：att_α = baseline + α * (actual - baseline)
            att_interpolated = att_baseline + alpha * (att_actual - att_baseline)
            att_interpolated = att_interpolated.detach().requires_grad_(True)
            
            # Forward pass 得到 logits（使用 block 结束时的 input_ids）
            logits = self._forward_with_layer_head_cache(
                input_ids=actual_input_ids,
                attention_mask=attention_mask,
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
        
        # 只在该 block 的生成位置（queries）上聚合，并只统计“本 block 真正从 [MASK] -> 非 [MASK]”的位置，
        # 避免 prompt/已确定 token 主导归因，也避免把仍是 [MASK] 的位置混进来导致大量无意义的负分。
        pos = torch.tensor(block_positions, device=elementwise_attribution.device, dtype=torch.long)
        pos_attr = elementwise_attribution[:, :, pos, :]  # (B, nh, |pos|, hs)
        changed = ((baseline_input_ids[:, pos] == mask_id) & (actual_input_ids[:, pos] != mask_id)).to(pos_attr.dtype)  # (B, |pos|)
        pos_attr = pos_attr * changed[:, None, :, None]

        # 对每个 head 在 batch, pos, head_dim 维度上求和
        attributions = pos_attr.sum(dim=[0, 2, 3])  # (nh,)

        if debug:
            # IG completeness sanity check in the injected variable space
            with torch.no_grad():
                logits_a0 = self._forward_with_layer_head_cache(
                    input_ids=actual_input_ids,
                    attention_mask=attention_mask,
                    target_layer_idx=target_layer_idx,
                    head_att_values=att_baseline,
                )
                logits_a1 = self._forward_with_layer_head_cache(
                    input_ids=actual_input_ids,
                    attention_mask=attention_mask,
                    target_layer_idx=target_layer_idx,
                    head_att_values=att_actual,
                )
                delta_f = float(_objective_from_logits(logits_a1) - _objective_from_logits(logits_a0))
                attr_sum = float(attributions.sum())
                denom = max(1e-12, abs(delta_f))
                rel_err = abs(attr_sum - delta_f) / denom
                print(
                    f"[ATTR-DEBUG][layer={target_layer_idx}] IG-check sum(attr)={attr_sum:+.6f} "
                    f"Δf={delta_f:+.6f} rel_err={rel_err:.3e}"
                )

            k = max(1, min(int(debug_topk), int(n_heads)))
            top_pos = torch.topk(attributions, k=k, largest=True)
            top_neg = torch.topk(attributions, k=k, largest=False)
            print(
                f"[ATTR-DEBUG][layer={target_layer_idx}] top+ heads="
                f"{list(zip(top_pos.indices.tolist(), [float(x) for x in top_pos.values]))}"
            )
            print(
                f"[ATTR-DEBUG][layer={target_layer_idx}] top- heads="
                f"{list(zip(top_neg.indices.tolist(), [float(x) for x in top_neg.values]))}"
            )
        
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

