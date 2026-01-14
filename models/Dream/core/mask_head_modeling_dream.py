"""
Dream head masking / pruning utilities.

目标：根据 head importance 分数，对 Dream 的 attention heads 做剪枝验证：
- prune_which="most":  剪掉全局最重要 top-k heads（跨 layer 全局排序）
- prune_which="least": 剪掉全局最不重要 top-k heads（跨 layer 全局排序）
- prune_which="random": 在指定 layer 范围内全局随机剪枝 top-k heads

实现方式：
1) 加载 importance_scores (layer_idx -> tensor[n_heads])
2) 生成 per-layer keep_mask (True=保留, False=剪掉)，注意：top-k 是“全局”而不是逐层
3) 就地 patch DreamAttention / DreamSdpaAttention 的 forward，在 (B, n_heads, T, head_dim) 阶段乘以 keep_mask
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch

__all__ = [
    "load_importance_scores_pt",
    "build_head_keep_masks_global",
    "build_random_head_keep_masks_global",
    "iter_dream_attn_modules",
    "patch_dream_attention_for_head_masking",
    "apply_head_keep_masks_",
]


def load_importance_scores_pt(pt_path: Union[str, Path]) -> Tuple[Dict[int, torch.Tensor], Dict[str, Any]]:
    """
    加载 head_importance.pt（Dream attribution 产物）：
      {
        "importance_scores": {layer_idx: tensor[n_heads], ...},
        "metadata": {...}
      }
    """
    pt_path = Path(pt_path)
    data = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    scores = data.get("importance_scores", None)
    if not isinstance(scores, dict) or len(scores) == 0:
        raise ValueError(f"Invalid or empty importance_scores in: {pt_path}")

    out: Dict[int, torch.Tensor] = {}
    for k, v in scores.items():
        lk = int(k)
        if not torch.is_tensor(v):
            v = torch.tensor(v)
        out[lk] = v.detach().to(torch.float32).clone()

    meta = data.get("metadata", {})
    if not isinstance(meta, dict):
        meta = {}
    return out, meta


def _resolve_layer_end(layer_end: int, n_layers: int) -> int:
    if layer_end < 0:
        return n_layers - 1
    return min(int(layer_end), n_layers - 1)


def build_head_keep_masks_global(
    importance_scores: Dict[int, torch.Tensor],
    *,
    prune_which: str,
    k: Optional[int] = None,
    k_frac: Optional[float] = None,
    layer_start: int = 0,
    layer_end: int = -1,
    keep_at_least_one_head: bool = True,
) -> Dict[int, torch.Tensor]:
    """
    基于 importance_scores 做 **跨 layer 全局** top-k 剪枝（most/least）。

    返回：dict[layer_idx] = bool tensor[n_heads]，True=保留，False=剪掉。
    """
    if prune_which not in {"most", "least"}:
        raise ValueError(f"prune_which must be 'most' or 'least', got: {prune_which}")
    if (k is None) == (k_frac is None):
        raise ValueError("Exactly one of k or k_frac must be provided.")
    if layer_start < 0:
        raise ValueError(f"layer_start must be >=0, got {layer_start}")

    layer_ids = sorted(int(x) for x in importance_scores.keys())
    if not layer_ids:
        raise ValueError("importance_scores is empty.")

    n_layers = max(layer_ids) + 1
    layer_end = _resolve_layer_end(layer_end, n_layers)
    if layer_start > layer_end:
        raise ValueError(f"Invalid layer range: {layer_start}..{layer_end}")

    # Init keep-all masks for all layers present in file
    keep_masks: Dict[int, torch.Tensor] = {
        int(li): torch.ones((int(importance_scores[int(li)].numel()),), dtype=torch.bool) for li in layer_ids
    }

    flat_scores = []
    flat_meta: list[tuple[int, int]] = []  # (layer_idx, head_idx)
    for li in layer_ids:
        scores = importance_scores[li].to(torch.float32).flatten()
        n_heads = int(scores.numel())
        if n_heads <= 0:
            raise ValueError(f"Layer {li} has invalid n_heads={n_heads}")
        if li < layer_start or li > layer_end:
            continue
        flat_scores.append(scores)
        flat_meta.extend([(li, hi) for hi in range(n_heads)])

    if not flat_scores:
        return keep_masks

    all_scores = torch.cat(flat_scores, dim=0)
    total_heads = int(all_scores.numel())
    if total_heads <= 0:
        return keep_masks

    if k is None:
        frac = float(k_frac)
        if not (0.0 <= frac <= 1.0):
            raise ValueError(f"k_frac must be in [0,1], got {frac}")
        k_i = int(round(total_heads * frac))
        k_i = max(1, k_i)
    else:
        k_i = int(k)

    if keep_at_least_one_head:
        k_i = min(k_i, total_heads - 1)
    else:
        k_i = min(k_i, total_heads)
    if k_i <= 0:
        return keep_masks

    largest = True if prune_which == "most" else False
    _, flat_prune_idx = torch.topk(all_scores, k=k_i, largest=largest)

    for idx in flat_prune_idx.tolist():
        li, hi = flat_meta[int(idx)]
        keep_masks[li][hi] = False

    return keep_masks


def build_random_head_keep_masks_global(
    *,
    n_layers: int,
    n_heads: int,
    k: Optional[int] = None,
    k_frac: Optional[float] = None,
    layer_start: int = 0,
    layer_end: int = -1,
    seed: int = 1234,
    keep_at_least_one_head: bool = True,
) -> Dict[int, torch.Tensor]:
    """
    Random baseline（global）：在指定 layer 范围内，把所有 heads 当成一个集合，
    一次性随机抽取 top-k 进行剪掉。
    """
    if (k is None) == (k_frac is None):
        raise ValueError("Exactly one of k or k_frac must be provided.")
    if n_layers <= 0 or n_heads <= 0:
        raise ValueError(f"Invalid n_layers/n_heads: n_layers={n_layers}, n_heads={n_heads}")
    if layer_start < 0:
        raise ValueError(f"layer_start must be >=0, got {layer_start}")

    layer_end = _resolve_layer_end(layer_end, n_layers)
    sel_layers = [li for li in range(n_layers) if layer_start <= li <= layer_end]
    total_heads = int(len(sel_layers) * n_heads)
    if total_heads <= 0:
        return {li: torch.ones((n_heads,), dtype=torch.bool) for li in range(n_layers)}

    if k is None:
        frac = float(k_frac)
        if not (0.0 <= frac <= 1.0):
            raise ValueError(f"k_frac must be in [0,1], got {frac}")
        k_i = int(round(total_heads * frac))
        k_i = max(1, k_i)
    else:
        k_i = int(k)

    if keep_at_least_one_head:
        k_i = min(k_i, total_heads - 1)
    else:
        k_i = min(k_i, total_heads)
    if k_i <= 0:
        return {li: torch.ones((n_heads,), dtype=torch.bool) for li in range(n_layers)}

    keep_masks: Dict[int, torch.Tensor] = {li: torch.ones((n_heads,), dtype=torch.bool) for li in range(n_layers)}

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    perm = torch.randperm(total_heads, generator=g)
    prune_flat = perm[:k_i].tolist()
    for p in prune_flat:
        li = sel_layers[int(p) // int(n_heads)]
        hi = int(p) % int(n_heads)
        keep_masks[li][hi] = False

    return keep_masks


def iter_dream_attn_modules(dream_model: torch.nn.Module) -> Iterable[torch.nn.Module]:
    """
    遍历模型中所有 DreamAttention 系列模块（包括 DreamSdpaAttention / DreamAttention）。
    """
    from models.Dream.core.modeling_dream import DreamAttention  # local import to avoid circulars

    for m in dream_model.modules():
        if isinstance(m, DreamAttention):
            yield m


def patch_dream_attention_for_head_masking(dream_model: torch.nn.Module) -> None:
    """
    就地 patch DreamAttention / DreamSdpaAttention.forward，在 head 维度应用 mask。
    mask 存在模块属性 `_head_keep_mask_q`：shape (n_heads,) float(0/1)。
    """
    from models.Dream.core.modeling_dream import DreamAttention, DreamSdpaAttention, apply_rotary_pos_emb, repeat_kv
    import math
    import torch.nn.functional as F

    # 1. 先定义通用的手动 attention 逻辑（带 mask）
    def _attn_forward_masked(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = torch.nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        # ===== Head mask insertion =====
        head_keep_mask = getattr(self, "_head_keep_mask_q", None)
        if head_keep_mask is not None:
            if not torch.is_tensor(head_keep_mask):
                head_keep_mask = torch.tensor(head_keep_mask, device=attn_output.device)
            head_keep_mask = head_keep_mask.to(device=attn_output.device, dtype=attn_output.dtype)
            attn_output = attn_output * head_keep_mask.view(1, -1, 1, 1)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value

    # 2. 遍历并应用 patch
    for attn in iter_dream_attn_modules(dream_model):
        if getattr(attn, "_head_mask_patched", False):
            continue

        if isinstance(attn, DreamSdpaAttention):
            # Copy of DreamSdpaAttention.forward with one-line head mask insertion.
            def _sdpa_forward_masked(
                self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value=None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: Optional[torch.LongTensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            ):
                if output_attentions:
                    # 如果需要 output_attentions，转而调用我们上面定义的手动带 mask 的逻辑
                    return _attn_forward_masked(self, hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache, cache_position, position_embeddings)

                bsz, q_len, _ = hidden_states.size()
                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                if position_embeddings is None:
                    cos, sin = self.rotary_emb(value_states, position_ids)
                else:
                    cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                if past_key_value is not None:
                    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
                    key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

                if query_states.device.type == "cuda" and attention_mask is not None:
                    query_states = query_states.contiguous()
                    key_states = key_states.contiguous()
                    value_states = value_states.contiguous()

                attn_output = F.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    attn_mask=attention_mask if isinstance(attention_mask, torch.Tensor) else None,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=False,
                )

                # ===== Head mask insertion (B, n_heads, T, head_dim) =====
                head_keep_mask = getattr(self, "_head_keep_mask_q", None)
                if head_keep_mask is not None:
                    if not torch.is_tensor(head_keep_mask):
                        head_keep_mask = torch.tensor(head_keep_mask, device=attn_output.device)
                    head_keep_mask = head_keep_mask.to(device=attn_output.device, dtype=attn_output.dtype)
                    attn_output = attn_output * head_keep_mask.view(1, -1, 1, 1)

                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.view(bsz, q_len, self.hidden_size)
                attn_output = self.o_proj(attn_output)
                return attn_output, None, past_key_value

            attn.forward = _sdpa_forward_masked.__get__(attn, attn.__class__)  # type: ignore[method-assign]

        elif isinstance(attn, DreamAttention):
            # 直接使用上面定义好的 _attn_forward_masked
            attn.forward = _attn_forward_masked.__get__(attn, attn.__class__)  # type: ignore[method-assign]

        attn._head_mask_patched = True  # type: ignore[attr-defined]


def apply_head_keep_masks_(
    dream_model: torch.nn.Module,
    keep_masks: Dict[int, torch.Tensor],
    *,
    device: Optional[torch.device] = None,
) -> None:
    """
    把 per-layer keep_mask 写入 attention 模块属性 `_head_keep_mask_q`（float 0/1）。
    """
    for attn in iter_dream_attn_modules(dream_model):
        li = int(getattr(attn, "layer_idx", -1))
        if li < 0:
            continue
        mask = keep_masks.get(li, None)
        if mask is None:
            mask = torch.ones((int(getattr(attn, "num_heads", 0) or 0),), dtype=torch.bool)
        if device is not None:
            mask = mask.to(device)
        attn._head_keep_mask_q = mask.to(torch.float32)  # type: ignore[attr-defined]

