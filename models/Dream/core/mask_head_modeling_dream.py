"""
Dream head masking / pruning utilities.

Pruning modes:
- prune_which="most":  prune globally highest-scoring top-k units
- prune_which="least": prune globally lowest-scoring top-k units
- prune_which="random": random pruning within specified layer range

Granularity (mask_granularity):
- "head":     per query-head (original behaviour)
- "kv_group": per KV-group (all query heads sharing a KV pair are pruned/kept
              together; importance averaged within group). Aligns with
              adaptive sparse ``gqa_weight_mode="kv"``.

Pipeline:
1) Load importance_scores  (layer_idx -> tensor[n_q_heads])
2) Build per-layer keep_mask (True=keep, False=prune) via global ranking
3) Monkey-patch DreamAttention forward to multiply attn_output by keep_mask
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_importance_scores_pt(pt_path: Union[str, Path]) -> Tuple[Dict[int, torch.Tensor], Dict[str, Any]]:
    """Load ``head_importance.pt`` produced by Dream attribution.

    Expected schema::

        {"importance_scores": {layer_idx: tensor[n_heads], ...}, "metadata": {...}}
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


def _to_kv_group_scores(per_head: torch.Tensor, n_kv_heads: int) -> torch.Tensor:
    """Average per-query-head importance into per-KV-group importance."""
    n_q = per_head.numel()
    gs = n_q // n_kv_heads
    return per_head.view(n_kv_heads, gs).mean(dim=1)


def _expand_group_mask(group_mask: torch.Tensor, group_size: int) -> torch.Tensor:
    """Expand per-KV-group bool mask to per-query-head bool mask."""
    return group_mask.repeat_interleave(group_size)


def _use_kv_group(n_kv_heads: Optional[int], n_q_heads: int) -> bool:
    """Determine whether KV-group granularity is active."""
    return (n_kv_heads is not None) and (0 < n_kv_heads < n_q_heads)


# ---------------------------------------------------------------------------
# Build masks (importance-based)
# ---------------------------------------------------------------------------

def build_head_keep_masks_global(
    importance_scores: Dict[int, torch.Tensor],
    *,
    prune_which: str,
    k: Optional[int] = None,
    k_frac: Optional[float] = None,
    layer_start: int = 0,
    layer_end: int = -1,
    keep_at_least_one_head: bool = True,
    prune_scope: str = "global",
    n_kv_heads: Optional[int] = None,
) -> Dict[int, torch.Tensor]:
    """Build keep-masks based on importance scores.

    When *n_kv_heads* is given and < n_q_heads, ranking and pruning operate at
    KV-group granularity: importance within each group is averaged, and all
    query heads in a pruned group are zeroed together.

    ``k`` / ``k_frac`` always refer to the number / fraction of **prunable
    units** (groups when using kv_group granularity, heads otherwise).

    Returns dict[layer_idx] -> bool tensor[n_q_heads].
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

    # Per-layer number of query heads (usually constant across layers)
    n_q_per_layer = {int(li): int(importance_scores[int(li)].numel()) for li in layer_ids}

    # Init keep-all masks for every layer in the file
    keep_masks: Dict[int, torch.Tensor] = {
        li: torch.ones((n_q_per_layer[li],), dtype=torch.bool) for li in layer_ids
    }

    largest = (prune_which == "most")

    # Decide granularity per layer
    def _layer_granularity(li: int) -> Tuple[torch.Tensor, int]:
        """Return (scores_for_ranking, group_size) for a layer."""
        raw = importance_scores[li].to(torch.float32).flatten()
        n_q = raw.numel()
        if _use_kv_group(n_kv_heads, n_q):
            return _to_kv_group_scores(raw, n_kv_heads), n_q // n_kv_heads  # type: ignore[arg-type]
        return raw, 1

    if prune_scope == "layer":
        for li in range(layer_start, layer_end + 1):
            if li not in importance_scores:
                continue
            scores, gs = _layer_granularity(li)
            n_units = int(scores.numel())
            if k is not None:
                k_i = min(int(k), n_units)
            else:
                k_i = max(1, int(round(n_units * float(k_frac))))
            if keep_at_least_one_head:
                k_i = min(k_i, n_units - 1)
            if k_i > 0:
                _, prune_idx = torch.topk(scores, k=k_i, largest=largest)
                for gi in prune_idx.tolist():
                    for qi in range(gi * gs, gi * gs + gs):
                        keep_masks[li][qi] = False
    else:
        # Global ranking across all selected layers
        flat_scores = []
        flat_meta: list[tuple[int, int, int]] = []  # (layer_idx, group_idx, group_size)
        # For KV-group pruning, cap global selection to at most 1 pruned group per layer.
        # This prevents over-concentrating pruning on early layers where score tails are
        # often heavy on both positive and negative sides.
        max_pruned_units_per_layer: Optional[int] = 1 if _use_kv_group(n_kv_heads, next(iter(n_q_per_layer.values()))) else None
        for li in layer_ids:
            if li < layer_start or li > layer_end:
                continue
            scores, gs = _layer_granularity(li)
            flat_scores.append(scores)
            flat_meta.extend([(li, gi, gs) for gi in range(int(scores.numel()))])

        if not flat_scores:
            return keep_masks

        all_scores = torch.cat(flat_scores, dim=0)
        total_units = int(all_scores.numel())

        if k is None:
            k_i = max(1, int(round(total_units * float(k_frac))))
        else:
            k_i = int(k)

        if keep_at_least_one_head:
            k_i = min(k_i, total_units - 1)
        else:
            k_i = min(k_i, total_units)

        if k_i > 0:
            if max_pruned_units_per_layer is None:
                _, flat_prune_idx = torch.topk(all_scores, k=k_i, largest=largest)
                selected = flat_prune_idx.tolist()
            else:
                # Enforce per-layer cap by selecting from globally ranked candidates.
                sorted_idx = torch.argsort(all_scores, descending=largest).tolist()
                selected: list[int] = []
                pruned_count_per_layer: Dict[int, int] = {}
                for idx in sorted_idx:
                    li, _, _ = flat_meta[int(idx)]
                    used = pruned_count_per_layer.get(li, 0)
                    if used >= max_pruned_units_per_layer:
                        continue
                    selected.append(int(idx))
                    pruned_count_per_layer[li] = used + 1
                    if len(selected) >= k_i:
                        break

            for idx in selected:
                li, gi, gs = flat_meta[int(idx)]
                for qi in range(gi * gs, gi * gs + gs):
                    keep_masks[li][qi] = False

    return keep_masks


# ---------------------------------------------------------------------------
# Build masks (random baseline)
# ---------------------------------------------------------------------------

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
    prune_scope: str = "global",
    n_kv_heads: Optional[int] = None,
) -> Dict[int, torch.Tensor]:
    """Random pruning baseline.

    When *n_kv_heads* is given and < *n_heads*, random selection operates at
    KV-group level so entire groups are pruned together.
    """
    if (k is None) == (k_frac is None):
        raise ValueError("Exactly one of k or k_frac must be provided.")

    n_layers_total = int(n_layers)
    n_q = int(n_heads)
    layer_end = _resolve_layer_end(layer_end, n_layers_total)
    keep_masks: Dict[int, torch.Tensor] = {
        li: torch.ones((n_q,), dtype=torch.bool) for li in range(n_layers_total)
    }

    use_group = _use_kv_group(n_kv_heads, n_q)
    n_units_per_layer = (n_kv_heads if use_group else n_q)  # type: ignore[arg-type]
    gs = (n_q // n_kv_heads) if use_group else 1  # type: ignore[operator]

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    if prune_scope == "layer":
        for li in range(layer_start, layer_end + 1):
            if k is not None:
                k_i = min(int(k), n_units_per_layer)
            else:
                k_i = max(1, int(round(n_units_per_layer * float(k_frac))))
            if keep_at_least_one_head:
                k_i = min(k_i, n_units_per_layer - 1)
            if k_i > 0:
                perm = torch.randperm(n_units_per_layer, generator=g)
                for ui in perm[:k_i].tolist():
                    for qi in range(ui * gs, ui * gs + gs):
                        keep_masks[li][qi] = False
    else:
        sel_layers = [li for li in range(n_layers_total) if layer_start <= li <= layer_end]
        total_units = len(sel_layers) * n_units_per_layer
        max_pruned_units_per_layer: Optional[int] = 1 if use_group else None

        if k is None:
            k_i = max(1, int(round(total_units * float(k_frac))))
        else:
            k_i = int(k)

        if keep_at_least_one_head:
            k_i = min(k_i, total_units - 1)
        else:
            k_i = min(k_i, total_units)

        if max_pruned_units_per_layer is not None:
            k_i = min(k_i, len(sel_layers) * max_pruned_units_per_layer)

        if k_i > 0:
            if max_pruned_units_per_layer is None:
                perm = torch.randperm(total_units, generator=g)
                selected = perm[:k_i].tolist()
            else:
                perm = torch.randperm(total_units, generator=g).tolist()
                selected = []
                pruned_count_per_layer: Dict[int, int] = {}
                for p in perm:
                    li = sel_layers[int(p) // n_units_per_layer]
                    used = pruned_count_per_layer.get(li, 0)
                    if used >= max_pruned_units_per_layer:
                        continue
                    selected.append(int(p))
                    pruned_count_per_layer[li] = used + 1
                    if len(selected) >= k_i:
                        break

            for p in selected:
                li = sel_layers[int(p) // n_units_per_layer]
                ui = int(p) % n_units_per_layer
                for qi in range(ui * gs, ui * gs + gs):
                    keep_masks[li][qi] = False

    return keep_masks


# ---------------------------------------------------------------------------
# Model patching
# ---------------------------------------------------------------------------

def iter_dream_attn_modules(dream_model: torch.nn.Module) -> Iterable[torch.nn.Module]:
    """Iterate over all DreamAttention-family modules in the model."""
    from models.Dream.core.modeling_dream import DreamAttention

    for m in dream_model.modules():
        if isinstance(m, DreamAttention):
            yield m


def patch_dream_attention_for_head_masking(dream_model: torch.nn.Module) -> None:
    """Monkey-patch DreamAttention / DreamSdpaAttention forward to apply head masks.

    The mask is stored per-module as ``_head_keep_mask_q``: shape ``(n_q_heads,)``
    with float 0/1 values.
    """
    from models.Dream.core.modeling_dream import DreamAttention, DreamSdpaAttention, apply_rotary_pos_emb, repeat_kv
    import math
    import torch.nn.functional as F

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

        # ===== Head mask =====
        apply_head_mask = True
        warmup_frac = getattr(self, "_head_mask_warmup_frac", None)
        if warmup_frac is not None:
            try:
                wf = float(warmup_frac)
            except Exception:
                wf = 0.0
            if wf > 0.0:
                now_step = getattr(self, "_head_mask_now_step", None)
                whole_steps = getattr(self, "_head_mask_whole_steps", None)
                if now_step is not None and whole_steps is not None:
                    try:
                        warmup_steps = int(float(whole_steps) * wf)
                        if int(now_step) < max(0, warmup_steps):
                            apply_head_mask = False
                    except Exception:
                        pass

        head_keep_mask = getattr(self, "_head_keep_mask_q", None)
        if apply_head_mask and head_keep_mask is not None:
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

    for attn in iter_dream_attn_modules(dream_model):
        if getattr(attn, "_head_mask_patched", False):
            continue

        if isinstance(attn, DreamSdpaAttention):
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

                # ===== Head mask (B, n_heads, T, head_dim) =====
                apply_head_mask = True
                warmup_frac = getattr(self, "_head_mask_warmup_frac", None)
                if warmup_frac is not None:
                    try:
                        wf = float(warmup_frac)
                    except Exception:
                        wf = 0.0
                    if wf > 0.0:
                        now_step = getattr(self, "_head_mask_now_step", None)
                        whole_steps = getattr(self, "_head_mask_whole_steps", None)
                        if now_step is not None and whole_steps is not None:
                            try:
                                warmup_steps = int(float(whole_steps) * wf)
                                if int(now_step) < max(0, warmup_steps):
                                    apply_head_mask = False
                            except Exception:
                                pass

                head_keep_mask = getattr(self, "_head_keep_mask_q", None)
                if apply_head_mask and head_keep_mask is not None:
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
            attn.forward = _attn_forward_masked.__get__(attn, attn.__class__)  # type: ignore[method-assign]

        attn._head_mask_patched = True  # type: ignore[attr-defined]

    try:
        dream_model._head_mask_attn_modules = list(iter_dream_attn_modules(dream_model))  # type: ignore[attr-defined]
    except Exception:
        pass


def apply_head_keep_masks_(
    dream_model: torch.nn.Module,
    keep_masks: Dict[int, torch.Tensor],
    *,
    device: Optional[torch.device] = None,
    head_mask_warmup_frac: Optional[float] = None,
) -> None:
    """Write per-layer keep_mask into each attention module as ``_head_keep_mask_q``."""
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

        if head_mask_warmup_frac is not None:
            attn._head_mask_warmup_frac = float(head_mask_warmup_frac)  # type: ignore[attr-defined]
