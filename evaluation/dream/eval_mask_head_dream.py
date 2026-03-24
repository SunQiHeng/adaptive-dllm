#!/usr/bin/env python3
"""
Dream head pruning / masking evaluation with lm-eval-harness.

Registered model:
  --model dream_mask_head_eval

--model_args:
- model_path=...
- prune_which=most|least|random
- prune_k=... or prune_k_frac=... (mutually exclusive)
- layer_start=0, layer_end=...
- random_prune_seed=...
- importance_path=/path/to/head_importance.pt  (not needed for random)
- mask_granularity=kv_group|head  (default: kv_group)

All other generation / MC params are inherited from DreamEvalHarness.
"""

from __future__ import annotations

import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model

from evaluation.dream.eval_dream import DreamEvalHarness, set_seed
from models.Dream.core.mask_head_modeling_dream import (
    apply_head_keep_masks_,
    build_head_keep_masks_global,
    build_random_head_keep_masks_global,
    load_importance_scores_pt,
    patch_dream_attention_for_head_masking,
)


def _str_to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).lower().strip()
    return s in ("1", "true", "t", "yes", "y")


def _strip_quotes(s: Any) -> str:
    if s is None:
        return ""
    t = str(s).strip()
    if len(t) >= 2 and ((t[0] == t[-1] == '"') or (t[0] == t[-1] == "'")):
        t = t[1:-1].strip()
    return t


@register_model("dream_mask_head_eval")
class DreamMaskHeadEvalHarness(DreamEvalHarness):
    """Head pruning verification on Dream standard attention."""

    def __init__(
        self,
        model_path="/data/qh_models/Dream-v0-Instruct-7B",
        importance_path=None,
        prune_which: str = "most",  # most|least|random
        prune_k=None,
        prune_k_frac=None,
        layer_start: int = 0,
        layer_end: int = -1,
        random_prune_seed: int = 1234,
        keep_at_least_one_head=True,
        head_mask_warmup_frac: float = 0.0,
        prune_scope: str = "global",
        mask_granularity: str = "kv_group",  # "kv_group" | "head"
        **kwargs,
    ):
        super().__init__(model_path=model_path, model_type="standard", **kwargs)

        prune_which = str(prune_which).strip()
        prune_scope = str(prune_scope).strip().lower()
        mask_granularity = str(mask_granularity).strip().lower()

        n_layers = int(getattr(self.model.config, "num_hidden_layers"))
        n_q_heads = int(getattr(self.model.config, "num_attention_heads"))
        n_kv_heads_cfg = int(getattr(self.model.config, "num_key_value_heads", n_q_heads))

        # Determine effective n_kv_heads for mask building
        n_kv_heads: int | None = None
        if mask_granularity == "kv_group" and n_kv_heads_cfg < n_q_heads:
            n_kv_heads = n_kv_heads_cfg
            group_size = n_q_heads // n_kv_heads
            print(f"[head_prune] mask_granularity=kv_group  n_q_heads={n_q_heads}  n_kv_heads={n_kv_heads}  group_size={group_size}")
        else:
            if mask_granularity == "kv_group" and n_kv_heads_cfg == n_q_heads:
                print(f"[head_prune] mask_granularity=kv_group requested but model is MHA (n_kv_heads==n_q_heads={n_q_heads}), falling back to per-head")
            print(f"[head_prune] mask_granularity=head  n_q_heads={n_q_heads}")

        meta = {}
        if prune_which == "random":
            keep_masks = build_random_head_keep_masks_global(
                n_layers=n_layers,
                n_heads=n_q_heads,
                k=int(prune_k) if prune_k is not None else None,
                k_frac=float(prune_k_frac) if prune_k_frac is not None else None,
                layer_start=int(layer_start),
                layer_end=int(layer_end),
                seed=int(random_prune_seed),
                keep_at_least_one_head=_str_to_bool(keep_at_least_one_head),
                prune_scope=prune_scope,
                n_kv_heads=n_kv_heads,
            )
        else:
            importance_path = _strip_quotes(importance_path)
            if not importance_path:
                raise ValueError("importance_path is required unless prune_which='random'")
            scores, meta = load_importance_scores_pt(importance_path)
            keep_masks = build_head_keep_masks_global(
                scores,
                prune_which=prune_which,
                k=int(prune_k) if prune_k is not None else None,
                k_frac=float(prune_k_frac) if prune_k_frac is not None else None,
                layer_start=int(layer_start),
                layer_end=int(layer_end),
                keep_at_least_one_head=_str_to_bool(keep_at_least_one_head),
                prune_scope=prune_scope,
                n_kv_heads=n_kv_heads,
            )

        patch_dream_attention_for_head_masking(self.model)
        try:
            dev = next(self.model.parameters()).device
        except StopIteration:
            dev = None
        apply_head_keep_masks_(
            self.model,
            keep_masks,
            device=dev,
            head_mask_warmup_frac=float(head_mask_warmup_frac),
        )

        # Print pruning summary
        total_pruned = 0
        total_heads = 0
        for li, m in sorted(keep_masks.items()):
            n = int(m.numel())
            pruned = int((~m).sum().item())
            total_pruned += pruned
            total_heads += n
        granularity_tag = f"kv_group(gs={n_q_heads // (n_kv_heads or n_q_heads)})" if n_kv_heads else "head"
        print(f"[head_prune] mode={prune_which}  granularity={granularity_tag}  pruned={total_pruned}/{total_heads} ({(total_pruned/max(1,total_heads))*100:.1f}%)")
        if meta:
            print(f"[head_prune] importance metadata keys: {sorted(list(meta.keys()))[:20]}")


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
