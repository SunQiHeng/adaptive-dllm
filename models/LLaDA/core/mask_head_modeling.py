"""
Head masking / pruning utilities for LLaDA.

目标：根据预先计算的 head importance 分数，对每一层的 attention heads 做“剪枝”验证：
- prune_most:  剪掉最重要的 top-k heads
- prune_least: 剪掉最不重要的 top-k heads

实现方式（不改权重、不改结构）：
1) 加载 importance_scores (layer_idx -> tensor[n_heads])
2) 生成 per-layer keep_mask (True=保留, False=剪掉)
3) 对模型的每个 block 就地 patch `attention()`：在 attention 输出的 head 维度乘以 keep_mask

这样可以直接在现有 `LLaDAModelLM` 上工作，避免重新实现/加载一套模型结构。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import torch

__all__ = [
    "HeadPruningSpec",
    "load_importance_scores_pt",
    "build_head_keep_masks",
    "build_random_head_keep_masks",
    "iter_llada_blocks",
    "patch_llada_blocks_for_head_masking",
    "apply_head_keep_masks_",
]


@dataclass(frozen=True)
class HeadPruningSpec:
    """
    描述一次 head 剪枝实验。

    - prune_which:
        - "most":  剪掉最重要的 heads (largest=True)
        - "least": 剪掉最不重要的 heads (largest=False)
    - k / k_frac: 二选一（k_frac 会按 n_heads * k_frac 四舍五入，并至少剪 1 个 head）
    """

    prune_which: str  # "most" | "least"
    k: Optional[int] = None
    k_frac: Optional[float] = None
    layer_start: int = 0
    layer_end: int = -1  # inclusive; -1 means last layer
    keep_at_least_one_head: bool = True


def load_importance_scores_pt(pt_path: Union[str, Path]) -> Tuple[Dict[int, torch.Tensor], Dict[str, Any]]:
    """
    加载 head importance 文件（通常是 head_importance.pt）。

    期望格式（你的项目里多处使用）：
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


def build_head_keep_masks(
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
    根据 importance_scores 生成 **global** keep_mask（跨 layer 全局排序/选择）。

    返回：dict[layer_idx] = bool tensor[n_heads]，True=保留，False=剪掉。

    说明：
    - “global”的含义是：在指定 layer 范围内，把所有 heads 视为一个集合做 top-k（most/least）。
    - 不再保证“每层剪掉固定比例/数量”。
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

    # Initialize keep masks to "keep all" for all layers we have scores for.
    keep_masks: Dict[int, torch.Tensor] = {}
    for layer_idx, scores in importance_scores.items():
        li = int(layer_idx)
        keep_masks[li] = torch.ones((int(scores.numel()),), dtype=torch.bool)

    # Collect all heads within layer range into one flat vector for global top-k.
    flat_scores = []
    flat_meta: list[tuple[int, int]] = []  # (layer_idx, head_idx)
    for li in sorted(int(x) for x in importance_scores.keys()):
        scores = importance_scores[li].to(torch.float32).flatten()
        n_heads = int(scores.numel())
        if n_heads <= 0:
            raise ValueError(f"Layer {li} has invalid n_heads={n_heads}")

        if li < layer_start or li > layer_end:
            continue

        flat_scores.append(scores)
        flat_meta.extend([(li, hi) for hi in range(n_heads)])

    if not flat_scores:
        # Nothing selected in the layer range -> return all-keep masks.
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

    # Apply pruning indices back to per-layer masks
    for idx in flat_prune_idx.tolist():
        li, hi = flat_meta[int(idx)]
        keep_masks[li][hi] = False

    return keep_masks


def build_random_head_keep_masks(
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
    随机剪枝 baseline（global）：在指定 layer 范围内，把所有 heads 视为一个集合，
    一次性随机选择 top-k 进行剪掉（均匀随机，不用 importance）。

    返回：dict[layer_idx] = bool tensor[n_heads]，True=保留，False=剪掉。
    """
    if (k is None) == (k_frac is None):
        raise ValueError("Exactly one of k or k_frac must be provided.")
    if n_layers <= 0 or n_heads <= 0:
        raise ValueError(f"Invalid n_layers/n_heads: n_layers={n_layers}, n_heads={n_heads}")
    if layer_start < 0:
        raise ValueError(f"layer_start must be >=0, got {layer_start}")

    layer_end = _resolve_layer_end(layer_end, n_layers)
    if layer_start > layer_end:
        raise ValueError(f"Invalid layer range: {layer_start}..{layer_end}")

    # total heads within selected layer range
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

    # Sample k_i heads globally (without replacement) from the flattened [layer, head] space.
    perm = torch.randperm(total_heads, generator=g)
    prune_flat = perm[:k_i].tolist()

    for p in prune_flat:
        li = sel_layers[int(p) // int(n_heads)]
        hi = int(p) % int(n_heads)
        keep_masks[li][hi] = False

    return keep_masks


def iter_llada_blocks(llada_lm: torch.nn.Module) -> Iterable[torch.nn.Module]:
    """
    遍历 LLaDA 模型的每一层 block（兼容 block_group_size==1 或 >1）。

    兼容输入：
    - `LLaDAModelLM`（HF wrapper），其内部是 `.model`
    - `LLaDAModel`（裸模型），其内部有 `.transformer`
    """
    m = getattr(llada_lm, "model", llada_lm)
    transformer = getattr(m, "transformer", None)
    if transformer is None:
        raise ValueError("Cannot find `.transformer` on the provided model.")

    if hasattr(transformer, "blocks"):
        for b in transformer.blocks:  # type: ignore[attr-defined]
            yield b
    elif hasattr(transformer, "block_groups"):
        for g in transformer.block_groups:  # type: ignore[attr-defined]
            for b in g:
                yield b
    else:
        raise ValueError("Transformer has neither `.blocks` nor `.block_groups`.")


def patch_llada_blocks_for_head_masking(llada_lm: torch.nn.Module) -> None:
    """
    就地 patch 每个 block 的 `attention()`，增加 head keep_mask 的乘法。

    说明：我们复制了 `models/LLaDA/core/modeling.py:LLaDABlock.attention` 的逻辑，
    只在 `att` (B, n_heads, T, head_dim) 阶段插入乘 mask。
    """
    for block in iter_llada_blocks(llada_lm):
        # 避免重复 patch
        if getattr(block, "_head_mask_patched", False):
            continue

        def _masked_attention(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attention_bias: Optional[torch.Tensor] = None,
            layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache: bool = False,
        ):
            B, T, C = q.size()
            dtype = k.dtype

            if getattr(self, "q_norm", None) is not None and getattr(self, "k_norm", None) is not None:
                q = self.q_norm(q).to(dtype=dtype)  # type: ignore[attr-defined]
                k = self.k_norm(k).to(dtype=dtype)  # type: ignore[attr-defined]

            q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
            k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
            v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

            if layer_past is not None:
                past_key, past_value = layer_past
                k = torch.cat((past_key, k), dim=-2)
                v = torch.cat((past_value, v), dim=-2)

            present = (k, v) if use_cache else None
            query_len, key_len = q.shape[-2], k.shape[-2]

            if getattr(self.config, "rope", False):
                q, k = self.rotary_emb(q, k)  # type: ignore[attr-defined]

            if attention_bias is not None:
                attention_bias = self._cast_attn_bias(  # type: ignore[attr-defined]
                    attention_bias[:, :, key_len - query_len : key_len, :key_len],
                    dtype,
                )

            att = self._scaled_dot_product_attention(  # type: ignore[attr-defined]
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0 if not self.training else self.config.attention_dropout,
                is_causal=False,
            )

            # ===== Head masking happens HERE (per-query-head) =====
            head_keep_mask = getattr(self, "_head_keep_mask_q", None)
            if head_keep_mask is not None:
                if not torch.is_tensor(head_keep_mask):
                    head_keep_mask = torch.tensor(head_keep_mask, device=att.device)
                head_keep_mask = head_keep_mask.to(device=att.device)
                if head_keep_mask.dtype != att.dtype:
                    head_keep_mask = head_keep_mask.to(dtype=att.dtype)
                # shape: (1, n_heads, 1, 1)
                att = att * head_keep_mask.view(1, -1, 1, 1)

            att = att.transpose(1, 2).contiguous().view(B, T, C)
            return self.attn_out(att), present  # type: ignore[attr-defined]

        # bind method
        block.attention = _masked_attention.__get__(block, block.__class__)  # type: ignore[method-assign]
        block._head_mask_patched = True  # type: ignore[attr-defined]


def apply_head_keep_masks_(
    llada_lm: torch.nn.Module,
    keep_masks: Dict[int, torch.Tensor],
    *,
    device: Optional[torch.device] = None,
) -> None:
    """
    把 keep_masks 写入每一层 block 的 `_head_keep_mask_q`（需要先 patch 才会生效）。
    """
    for block in iter_llada_blocks(llada_lm):
        layer_id = int(getattr(block, "layer_id", -1))
        if layer_id < 0:
            continue
        mask = keep_masks.get(layer_id, None)
        if mask is None:
            # 默认全保留
            mask = torch.ones((int(block.config.n_heads),), dtype=torch.bool)  # type: ignore[attr-defined]
        if device is not None:
            mask = mask.to(device)
        # 用 0/1 float mask 更省事（避免 bool * float 的隐式类型）
        block._head_keep_mask_q = mask.to(torch.float32)  # type: ignore[attr-defined]

